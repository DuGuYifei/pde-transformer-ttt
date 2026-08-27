from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PDEViTTTWindowBlock(nn.Module):
    """ViT^3-style local window mixer for PDE tokens.

    This is adapted from LeapLabTHU/ViTTT's TTT block, with the token interface
    kept local to one PDE window and periodic padding support for the 3x3 branch.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        inner_lr: float = 1.0,
        proj_drop: float = 0.0,
        padding_mode: str = "zero",
        key_instance_norm: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if padding_mode not in ("zero", "replicate"):
            raise ValueError("padding_mode must be 'zero' or 'replicate'")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inner_lr = inner_lr
        self.padding_mode = padding_mode
        self.key_instance_norm = key_instance_norm

        self.qkv = nn.Linear(dim, dim * 3 + self.head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, num_heads, self.head_dim, self.head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, num_heads, self.head_dim, self.head_dim))
        self.w3 = nn.Parameter(torch.zeros(self.head_dim, 1, 3, 3))
        self.proj = nn.Linear(dim + self.head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.scale = 9 ** -0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)
        nn.init.trunc_normal_(self.w3, std=0.02)

    def inner_train_simplified_swiglu(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        lr: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = k @ w1
        z2 = k @ w2
        gate = F.silu(z2)

        # Closed-form gradients follow LeapLabTHU/ViTTT's simplified SwiGLU
        # inner update, avoiding autograd over a per-head vector loss.
        grad_output = -v / float(v.shape[2]) * self.scale
        sigmoid_z2 = torch.sigmoid(z2)
        silu_grad = sigmoid_z2 * (1.0 + z2 * (1.0 - sigmoid_z2))

        grad_w1 = k.transpose(-2, -1) @ (grad_output * gate)
        grad_w2 = k.transpose(-2, -1) @ (grad_output * z1 * silu_grad)

        grad_w1 = grad_w1 / (grad_w1.norm(dim=(-2, -1), keepdim=True) + 1.0)
        grad_w2 = grad_w2 / (grad_w2.norm(dim=(-2, -1), keepdim=True) + 1.0)
        return w1 - lr * grad_w1, w2 - lr * grad_w2

    def _pad_2d(self, x: torch.Tensor, periodic: bool, padding_mode: str) -> torch.Tensor:
        if periodic:
            return F.pad(x, (1, 1, 1, 1), mode="circular")
        if padding_mode == "replicate":
            return F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.pad(x, (1, 1, 1, 1))

    def inner_train_3x3dwc(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        lr: float,
        periodic: bool = False,
        padding_mode: str = "zero",
    ) -> torch.Tensor:
        batch, channels, height, width = k.shape
        grad_output = -v / float(height * width) * self.scale
        k_padded = self._pad_2d(k, periodic=periodic, padding_mode=padding_mode)

        grads = []
        for dy in range(3):
            for dx in range(3):
                patch = k_padded[:, :, dy : dy + height, dx : dx + width]
                grads.append((patch * grad_output).sum(dim=(-2, -1)))
        grad_w = torch.stack(grads, dim=-1).reshape(batch * channels, 1, 3, 3)
        grad_w = grad_w / (grad_w.norm(dim=(-2, -1), keepdim=True) + 1.0)
        return w.repeat(batch, 1, 1, 1) - lr * grad_w

    def _depthwise_conv(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        periodic: bool,
        padding_mode: str,
    ) -> torch.Tensor:
        batch, channels, height, width = x.shape
        x = x.reshape(1, batch * channels, height, width)
        if periodic or padding_mode == "replicate":
            x = self._pad_2d(x, periodic=periodic, padding_mode=padding_mode)
            return F.conv2d(x, weight, padding=0, groups=batch * channels)
        return F.conv2d(x, weight, padding=1, groups=batch * channels)

    def forward(self, x: torch.Tensor, h: int, w: int, periodic: bool = False) -> torch.Tensor:
        batch, num_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {dim}")
        if num_tokens != h * w:
            raise ValueError(f"expected h*w tokens ({h * w}), got {num_tokens}")

        head_dim = self.head_dim
        q1, k1, v1, q2, k2, v2 = torch.split(
            self.qkv(x),
            [dim, dim, dim, head_dim, head_dim, head_dim],
            dim=-1,
        )

        q1 = q1.reshape(batch, num_tokens, self.num_heads, head_dim).transpose(1, 2)
        k1 = k1.reshape(batch, num_tokens, self.num_heads, head_dim).transpose(1, 2)
        v1 = v1.reshape(batch, num_tokens, self.num_heads, head_dim).transpose(1, 2)
        if self.key_instance_norm:
            k1_mean = k1.mean(dim=2, keepdim=True)
            k1_var = k1.var(dim=2, keepdim=True, unbiased=False)
            k1 = (k1 - k1_mean) * torch.rsqrt(k1_var + 1e-6)
        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2, lr=self.inner_lr)
        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(batch, num_tokens, dim)

        q2 = q2.reshape(batch, h, w, head_dim).permute(0, 3, 1, 2)
        k2 = k2.reshape(batch, h, w, head_dim).permute(0, 3, 1, 2)
        v2 = v2.reshape(batch, h, w, head_dim).permute(0, 3, 1, 2)
        if self.key_instance_norm:
            k2 = F.instance_norm(k2)
        w3 = self.inner_train_3x3dwc(
            k2,
            v2,
            self.w3,
            lr=self.inner_lr,
            periodic=periodic,
            padding_mode=self.padding_mode,
        )
        x2 = self._depthwise_conv(q2, w3, periodic=periodic, padding_mode=self.padding_mode)
        x2 = x2.reshape(batch, head_dim, num_tokens).transpose(1, 2)

        output = torch.cat([x1, x2], dim=-1)
        output = self.proj(output)
        return self.proj_drop(output)
