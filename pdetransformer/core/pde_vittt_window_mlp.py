import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class WindowFullBatchMLPTTTMixer(nn.Module):
    """Closed-form MLP TTT with optional sequential window schedules."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        inner_lr: float = 1.0,
        hidden_ratio: int = 4,
        update_mode: str = "full_batch",
        chunk_size: int = 16,
    ):
        super().__init__()
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        if hidden_ratio <= 0:
            raise ValueError(f"hidden_ratio must be positive, got {hidden_ratio}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.hidden_dim = hidden_ratio * self.head_dim
        self.inner_lr = inner_lr
        self.scale = 9**-0.5
        self.update_mode = update_mode
        self.chunk_size = chunk_size

        valid_modes = {"full_batch", "token_sequential", "window_sequential"}
        if update_mode not in valid_modes:
            raise ValueError(
                f"Unsupported update_mode={update_mode!r}; "
                f"expected one of {sorted(valid_modes)}."
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(
            torch.zeros(1, num_heads, self.head_dim, self.hidden_dim)
        )
        self.w2 = nn.Parameter(
            torch.zeros(1, num_heads, self.hidden_dim, self.head_dim)
        )
        self.reset_ttt_parameters()

    def reset_ttt_parameters(self) -> None:
        """Restore the experiment-specific initialization after PDE initialization."""
        trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        trunc_normal_(self.w1, std=0.02)
        trunc_normal_(self.w2, std=0.02)

    @staticmethod
    def _gelu_derivative(x: torch.Tensor) -> torch.Tensor:
        """Derivative of PyTorch's exact GELU implementation."""
        return 0.5 * (1.0 + torch.erf(x / 2.0**0.5)) + (
            x * torch.exp(-0.5 * x.square()) / (2.0 * torch.pi) ** 0.5
        )

    @staticmethod
    def _normalize_columns(gradient: torch.Tensor) -> torch.Tensor:
        return gradient / (gradient.norm(dim=-2, keepdim=True) + 1.0)

    def inner_train(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        initial_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one inner update to the supplied per-sample MLP weights."""
        token_count = k.shape[-2]
        if initial_weights is None:
            w1 = self.w1.to(dtype=k.dtype)
            w2 = self.w2.to(dtype=k.dtype)
        else:
            w1, w2 = initial_weights

        z1 = k @ w1
        hidden = F.gelu(z1)
        output_gradient = -(self.scale / float(token_count)) * v
        gradient_w2 = hidden.transpose(-2, -1) @ output_gradient
        gradient_hidden = output_gradient @ w2.transpose(-2, -1)
        gradient_w1 = k.transpose(-2, -1) @ (
            gradient_hidden * self._gelu_derivative(z1)
        )

        gradient_w1 = self._normalize_columns(gradient_w1)
        gradient_w2 = self._normalize_columns(gradient_w2)
        return (
            w1 - self.inner_lr * gradient_w1,
            w2 - self.inner_lr * gradient_w2,
        )

    @staticmethod
    def _apply_fast_weights(
        q: torch.Tensor,
        fast_weights: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fast_w1, fast_w2 = fast_weights
        return F.gelu(q @ fast_w1) @ fast_w2

    def _token_sequential(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, token_count, _ = q.shape
        fast_weights = (
            self.w1.to(dtype=q.dtype).expand(batch, -1, -1, -1),
            self.w2.to(dtype=q.dtype).expand(batch, -1, -1, -1),
        )
        outputs = []
        for start in range(0, token_count, self.chunk_size):
            stop = min(start + self.chunk_size, token_count)
            fast_weights = self.inner_train(
                k[..., start:stop, :],
                v[..., start:stop, :],
                initial_weights=fast_weights,
            )
            outputs.append(
                self._apply_fast_weights(q[..., start:stop, :], fast_weights)
            )
        return torch.cat(outputs, dim=-2)

    def _window_sequential(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        windows_per_sample: int,
    ) -> torch.Tensor:
        window_batch, num_heads, token_count, head_dim = q.shape
        if windows_per_sample <= 0 or window_batch % windows_per_sample != 0:
            raise ValueError(
                f"window batch {window_batch} must be divisible by "
                f"windows_per_sample={windows_per_sample}."
            )
        sample_batch = window_batch // windows_per_sample
        q = q.reshape(
            sample_batch, windows_per_sample, num_heads, token_count, head_dim
        )
        k = k.reshape_as(q)
        v = v.reshape_as(q)
        fast_weights = (
            self.w1.to(dtype=q.dtype).expand(sample_batch, -1, -1, -1),
            self.w2.to(dtype=q.dtype).expand(sample_batch, -1, -1, -1),
        )
        outputs = []
        for window_index in range(windows_per_sample):
            fast_weights = self.inner_train(
                k[:, window_index],
                v[:, window_index],
                initial_weights=fast_weights,
            )
            outputs.append(
                self._apply_fast_weights(q[:, window_index], fast_weights)
            )
        return torch.stack(outputs, dim=1).reshape(
            window_batch, num_heads, token_count, head_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
        periodic: bool = False,
        windows_per_sample: int = 1,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected [B, N, {self.dim}] input, got {tuple(x.shape)}."
            )
        if x.shape[1] != height * width:
            raise ValueError(
                f"Expected {height * width} window tokens, got {x.shape[1]}."
            )
        del periodic  # The inherited shifted-window path handles PDE boundaries.

        batch, token_count, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.update_mode == "full_batch":
            fast_weights = self.inner_train(k, v)
            output = self._apply_fast_weights(q, fast_weights)
        elif self.update_mode == "token_sequential":
            output = self._token_sequential(q, k, v)
        else:
            output = self._window_sequential(
                q, k, v, windows_per_sample=windows_per_sample
            )
        return output.transpose(1, 2).reshape(batch, token_count, self.dim)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, hidden_dim={self.hidden_dim}, "
            f"inner_lr={self.inner_lr}, update_mode={self.update_mode!r}, "
            f"chunk_size={self.chunk_size}"
        )
