import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class WindowFullBatchMLPTTTMixer(nn.Module):
    """One non-causal MLP fast-weight update over all tokens in one window."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        inner_lr: float = 1.0,
        hidden_ratio: int = 4,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct independent per-sample MLP weights from one full-window update."""
        token_count = k.shape[-2]
        w1 = self.w1.to(dtype=k.dtype)
        w2 = self.w2.to(dtype=k.dtype)

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
        fast_w1, fast_w2 = self.inner_train(k, v)
        output = F.gelu(q @ fast_w1) @ fast_w2
        return output.transpose(1, 2).reshape(batch, token_count, self.dim)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, hidden_dim={self.hidden_dim}, "
            f"inner_lr={self.inner_lr}"
        )
