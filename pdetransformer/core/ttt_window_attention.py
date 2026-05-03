from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.utils._pytree import tree_map


@dataclass
class TTTWindowConfig:
    hidden_size: int
    num_attention_heads: int
    ttt_layer_type: str = "linear"
    ttt_base_lr: float = 1.0
    mini_batch_size: int = 16
    rope_theta: float = 10000.0
    use_gate: bool = False
    scan_checkpoint_group_size: int = 0


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def permute_qk(q, k):
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    return q, k


def undo_permute_qk(q, k):
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    return q, k


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=16, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def scan(f, init, xs, out, checkpoint_group=0):
    carry = init
    num_items = len(next(iter(xs.values()))) if isinstance(xs, dict) else len(xs[0])

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry

    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out


def ln_fwd(x, gamma, beta, eps=1e-6):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    return gamma * ((x - mu) / std) + beta


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    dim = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    return (
        (1.0 / dim)
        * (
            dim * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )


def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)


class TTTBase(nn.Module):
    def __init__(self, config: TTTWindowConfig):
        super().__init__()
        self.config = config
        self.width = config.hidden_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.mini_batch_size = config.mini_batch_size

        if self.width % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("TTT rotary embedding requires an even head dimension")

        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))

        self.q_proj = nn.Linear(self.width, self.width, bias=False)
        self.k_proj = nn.Linear(self.width, self.width, bias=False)
        self.v_proj = nn.Linear(self.width, self.width, bias=False)
        self.o_proj = nn.Linear(self.width, self.width, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.mini_batch_size,
            base=config.rope_theta,
        )

        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )

        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

        self.use_gate = config.use_gate
        if self.use_gate:
            self.g_proj = nn.Linear(self.width, self.width, bias=False)
        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    def get_eta(self, x, mini_batch_size):
        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", x, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1, 1
        )
        ttt_lr = F.sigmoid(ttt_lr)
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.config.ttt_base_lr * ttt_lr / self.head_dim

        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[:mini_batch_size]
        token_idx = torch.clamp_min(token_idx, 0.0)
        token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (x.shape[0], self.num_heads, x.shape[1], mini_batch_size, 1),
        )
        return token_eta, ttt_lr_eta

    def apply_gate(self, hidden_states, ttt_output):
        y = self.g_proj(hidden_states)
        y = F.gelu(y, approximate="tanh")
        return y * ttt_output

    def get_ttt_inputs(self, inputs, mini_batch_size):
        xq = inputs["XQ"]
        xk = inputs["XK"]
        xv = inputs["XV"]
        x = inputs["X"]
        batch, seq_len, _ = x.shape
        num_mini_batch = seq_len // mini_batch_size

        x = x.reshape(batch, num_mini_batch, mini_batch_size, self.width)
        xq = xq.reshape(batch, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        xk = xk.reshape(batch, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        xv = xv.reshape(batch, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)

        token_eta, ttt_lr_eta = self.get_eta(x, mini_batch_size)
        eta = token_eta * ttt_lr_eta
        return {
            "XQ": xq,
            "XK": xk,
            "XV": xv,
            "eta": eta,
            "token_eta": token_eta,
            "ttt_lr_eta": ttt_lr_eta,
        }

    def ttt(self, inputs, mini_batch_size, last_mini_batch_params_dict):
        raise NotImplementedError

    def forward(self, hidden_states: torch.Tensor):
        batch, seq_len = hidden_states.shape[:2]
        reminder_len = seq_len % self.mini_batch_size
        num_mini_batch = seq_len // self.mini_batch_size
        last_mini_batch_params_dict = None

        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xv = xv.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(xv, position_ids % self.mini_batch_size)
        xq, xk = permute_qk(xq, xk)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        xq, xk = undo_permute_qk(xq, xk)

        output_hidden_states = []
        if num_mini_batch > 0:
            inputs = {
                "XQ": xq[:, :, : num_mini_batch * self.mini_batch_size],
                "XK": xk[:, :, : num_mini_batch * self.mini_batch_size],
                "XV": xv[:, :, : num_mini_batch * self.mini_batch_size],
                "X": hidden_states[:, : num_mini_batch * self.mini_batch_size],
            }
            output_mod, last_mini_batch_params_dict = self.ttt(
                self.get_ttt_inputs(inputs, self.mini_batch_size),
                mini_batch_size=self.mini_batch_size,
                last_mini_batch_params_dict=last_mini_batch_params_dict,
            )
            output_hidden_states.append(output_mod)

        if reminder_len > 0:
            inputs = {
                "XQ": xq[:, :, -reminder_len:],
                "XK": xk[:, :, -reminder_len:],
                "XV": xv[:, :, -reminder_len:],
                "X": hidden_states[:, -reminder_len:],
            }
            output_reminder, _ = self.ttt(
                self.get_ttt_inputs(inputs, reminder_len),
                mini_batch_size=reminder_len,
                last_mini_batch_params_dict=last_mini_batch_params_dict,
            )
            output_hidden_states.append(output_reminder)

        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.post_norm(output_hidden_states)
        if self.use_gate:
            output_hidden_states = self.apply_gate(hidden_states, output_hidden_states)
        return self.o_proj(output_hidden_states)


class TTTLinear(TTTBase):
    def __init__(self, config: TTTWindowConfig):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(self, inputs, mini_batch_size, last_mini_batch_params_dict):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        batch = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        seq_len = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype
        use_dual_form = mini_batch_size % self.mini_batch_size == 0

        def compute_mini_batch(params_dict, inputs):
            W1_init = params_dict["W1_states"]
            b1_init = params_dict["b1_states"]
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            x1 = XK_mini_batch
            z1 = x1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch
            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_z1 = ln_fused_l2_bwd(z1, reconstruction_target, ln_weight, ln_bias)

            if use_dual_form:
                attn1 = torch.tril(XQ_mini_batch @ x1.transpose(-2, -1))
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_z1
                z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * attn1) @ grad_l_wrt_z1 + b1_bar
                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                W1_last = W1_init - (last_eta_mini_batch * x1).transpose(-1, -2) @ grad_l_wrt_z1
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_z1, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (*ttt_lr_eta_mini_batch.shape[:2], mini_batch_size, mini_batch_size),
                )
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", x1, grad_l_wrt_z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_z1)
                grad_b1 = grad_b1 + params_dict["b1_grad"]
                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch
                z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]

            z1_bar = ln_fwd(z1_bar, ln_weight, ln_bias)
            XQW_mini_batch = XQ_mini_batch + z1_bar
            return {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
            }, XQW_mini_batch

        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(batch, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(batch, 1, 1, 1)),
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))

        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)
        XQW_batch = torch.empty(
            (num_mini_batch, batch, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            self.config.scan_checkpoint_group_size if self.training else 0,
        )
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        XQW_batch = XQW_batch.reshape(batch, seq_len, self.width)
        return XQW_batch, batch_params_dict


class TTTMLP(TTTBase):
    def __init__(self, config: TTTWindowConfig):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(self, inputs, mini_batch_size, last_mini_batch_params_dict):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        batch = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        seq_len = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype
        use_dual_form = mini_batch_size % self.mini_batch_size == 0

        def compute_mini_batch(params_dict, inputs):
            W1_init = params_dict["W1_states"]
            b1_init = params_dict["b1_states"]
            W2_init = params_dict["W2_states"]
            b2_init = params_dict["b2_states"]
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            x1 = XK_mini_batch
            z1 = x1 @ W1_init + b1_init
            x2 = F.gelu(z1, approximate="tanh")
            z2 = x2 @ W2_init + b2_init
            reconstruction_target = XV_mini_batch - XK_mini_batch
            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_z2 = ln_fused_l2_bwd(z2, reconstruction_target, ln_weight, ln_bias)
            grad_l_wrt_z1 = grad_l_wrt_z2 @ W2_init.transpose(-2, -1) * gelu_bwd(z1)

            if use_dual_form:
                attn1 = torch.tril(XQ_mini_batch @ x1.transpose(-2, -1))
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_z1
                z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * attn1) @ grad_l_wrt_z1 + b1_bar
                x2_bar = F.gelu(z1_bar, approximate="tanh")
                attn2 = torch.tril(x2_bar @ x2.transpose(-2, -1))
                b2_bar = b2_init - torch.tril(eta_mini_batch) @ grad_l_wrt_z2
                z2_bar = x2_bar @ W2_init - (eta_mini_batch * attn2) @ grad_l_wrt_z2 + b2_bar
                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                W1_last = W1_init - (last_eta_mini_batch * x1).transpose(-1, -2) @ grad_l_wrt_z1
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_z1, dim=-2, keepdim=True)
                W2_last = W2_init - (last_eta_mini_batch * x2).transpose(-1, -2) @ grad_l_wrt_z2
                b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_z2, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
                grad_W2_last = torch.zeros_like(W2_last)
                grad_b2_last = torch.zeros_like(b2_last)
            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (*ttt_lr_eta_mini_batch.shape[:2], mini_batch_size, mini_batch_size),
                )
                grad_W2 = torch.einsum("bhki,bhkj->bhkij", x2, grad_l_wrt_z2)
                grad_W2 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W2)
                grad_W2 = grad_W2 + params_dict["W2_grad"].unsqueeze(2)
                grad_b2 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_z2)
                grad_b2 = grad_b2 + params_dict["b2_grad"]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", x1, grad_l_wrt_z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_z1)
                grad_b1 = grad_b1 + params_dict["b1_grad"]

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch
                W2_bar = W2_init.unsqueeze(2) - grad_W2 * token_eta_mini_batch.unsqueeze(-1)
                b2_bar = b2_init - grad_b2 * token_eta_mini_batch
                z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
                x2_bar = F.gelu(z1_bar, approximate="tanh")
                z2_bar = (x2_bar.unsqueeze(3) @ W2_bar).squeeze(3) + b2_bar
                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                W2_last = W2_bar[:, :, -1]
                b2_last = b2_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]
                grad_W2_last = grad_W2[:, :, -1]
                grad_b2_last = grad_b2[:, :, -1:]

            z2_bar = ln_fwd(z2_bar, ln_weight, ln_bias)
            XQW_mini_batch = XQ_mini_batch + z2_bar
            return {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W2_states": W2_last,
                "b2_states": b2_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
                "W2_grad": grad_W2_last,
                "b2_grad": grad_b2_last,
            }, XQW_mini_batch

        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(batch, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(batch, 1, 1, 1)),
                "W2_states": torch.tile(self.W2.unsqueeze(0), dims=(batch, 1, 1, 1)),
                "b2_states": torch.tile(self.b2.unsqueeze(0), dims=(batch, 1, 1, 1)),
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))
            init_params_dict.update(W2_grad=torch.zeros_like(init_params_dict["W2_states"]))
            init_params_dict.update(b2_grad=torch.zeros_like(init_params_dict["b2_states"]))

        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)
        XQW_batch = torch.empty(
            (num_mini_batch, batch, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            self.config.scan_checkpoint_group_size if self.training else 0,
        )
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        XQW_batch = XQW_batch.reshape(batch, seq_len, self.width)
        return XQW_batch, batch_params_dict


class TTTWindowAttention2DTime(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        ttt_layer_type="linear",
        ttt_base_lr=1.0,
        mini_batch_size=16,
        use_gate=False,
        scan_checkpoint_group_size=0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            unsupported = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unsupported TTTWindowAttention2DTime arguments: {unsupported}")

        config = TTTWindowConfig(
            hidden_size=dim,
            num_attention_heads=num_heads,
            ttt_layer_type=ttt_layer_type,
            ttt_base_lr=ttt_base_lr,
            mini_batch_size=mini_batch_size,
            use_gate=use_gate,
            scan_checkpoint_group_size=scan_checkpoint_group_size,
        )
        if ttt_layer_type == "linear":
            self.ttt = TTTLinear(config)
        elif ttt_layer_type == "mlp":
            self.ttt = TTTMLP(config)
        else:
            raise ValueError(f"Invalid ttt_layer_type: {ttt_layer_type}")

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        if attn_mask is not None:
            raise NotImplementedError(
                "TTTWindowAttention2DTime does not support shifted-window masks yet; use periodic=True."
            )

        if x.dim() == 3:
            return self.ttt(x)
        if x.dim() == 4:
            batch, num_channels, num_tokens, dim = x.shape
            x = x.reshape(batch * num_channels, num_tokens, dim)
            x = self.ttt(x)
            return x.reshape(batch, num_channels, num_tokens, dim)
        raise ValueError(
            "TTTWindowAttention2DTime expects input shape (B, N, C) or (B, D, N, C)."
        )
