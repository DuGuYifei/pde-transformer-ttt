import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class GlobalLinearTTTMixer(nn.Module):
    """One-step non-causal linear TTT over a complete PDE feature map."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        inner_lr: float = 1.0,
    ):
        super().__init__()
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inner_lr = inner_lr
        self.scale = 9**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.w0 = nn.Parameter(
            torch.zeros(1, self.num_heads, self.head_dim, self.head_dim)
        )
        self.reset_ttt_parameters()

    def reset_ttt_parameters(self) -> None:
        """Restore the ViT-style initialization after the PDE-wide initializer."""
        trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        trunc_normal_(self.w0, std=0.02)

    def inner_train(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Return independent per-sample fast weights after one inner update."""
        token_count = k.shape[-2]
        gradient = -(self.scale / float(token_count)) * (
            k.transpose(-2, -1) @ v
        )
        # Match ViT^3's stabilization: normalize each output column separately.
        gradient = gradient / (gradient.norm(dim=-2, keepdim=True) + 1.0)
        initial_weights = self.w0.to(dtype=gradient.dtype)
        return initial_weights - self.inner_lr * gradient

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
        periodic: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected [B, N, {self.dim}] input, got {tuple(x.shape)}."
            )
        if x.shape[1] != height * width:
            raise ValueError(
                f"Expected {height * width} spatial tokens, got {x.shape[1]}."
            )
        del periodic  # Boundary handling belongs to the full-map CPE.

        batch, token_count, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        fast_weights = self.inner_train(k, v)
        output = q @ fast_weights
        return output.transpose(1, 2).reshape(batch, token_count, self.dim)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, inner_lr={self.inner_lr}"
        )
