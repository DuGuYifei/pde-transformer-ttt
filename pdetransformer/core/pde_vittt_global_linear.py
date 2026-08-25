import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class GlobalLinearTTTMixer(nn.Module):
    """Closed-form linear TTT with optional sequential window schedules."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        inner_lr: float = 1.0,
        update_mode: str = "full_batch",
        chunk_size: int = 16,
    ):
        super().__init__()
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
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

    def inner_train(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        initial_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one inner update to the supplied per-sample fast weights."""
        token_count = k.shape[-2]
        gradient = -(self.scale / float(token_count)) * (
            k.transpose(-2, -1) @ v
        )
        # Match ViT^3's stabilization: normalize each output column separately.
        gradient = gradient / (gradient.norm(dim=-2, keepdim=True) + 1.0)
        if initial_weights is None:
            initial_weights = self.w0.to(dtype=gradient.dtype)
        return initial_weights - self.inner_lr * gradient

    def _token_sequential(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, token_count, _ = q.shape
        fast_weights = self.w0.to(dtype=q.dtype).expand(batch, -1, -1, -1)
        outputs = []
        for start in range(0, token_count, self.chunk_size):
            stop = min(start + self.chunk_size, token_count)
            fast_weights = self.inner_train(
                k[..., start:stop, :],
                v[..., start:stop, :],
                initial_weights=fast_weights,
            )
            outputs.append(q[..., start:stop, :] @ fast_weights)
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
        fast_weights = self.w0.to(dtype=q.dtype).expand(
            sample_batch, -1, -1, -1
        )
        outputs = []
        for window_index in range(windows_per_sample):
            fast_weights = self.inner_train(
                k[:, window_index],
                v[:, window_index],
                initial_weights=fast_weights,
            )
            outputs.append(q[:, window_index] @ fast_weights)
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
        if self.update_mode == "full_batch":
            fast_weights = self.inner_train(k, v)
            output = q @ fast_weights
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
            f"head_dim={self.head_dim}, inner_lr={self.inner_lr}, "
            f"update_mode={self.update_mode!r}, chunk_size={self.chunk_size}"
        )
