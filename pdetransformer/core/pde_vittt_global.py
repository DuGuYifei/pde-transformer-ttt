import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class DepthwiseCPE2D(nn.Module):
    """Per-block conditional positional encoding on the complete feature map."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, periodic: bool) -> torch.Tensor:
        padding_mode = "circular" if periodic else "constant"
        return self.proj(F.pad(x, (1, 1, 1, 1), mode=padding_mode))

    def reset_official_parameters(self) -> None:
        # Official ViTTT leaves CPE convolutions at the PyTorch Conv2d default.
        self.proj.reset_parameters()


class RotaryEmbedding2D(nn.Module):
    """Resolution-independent implementation of the official H-ViT^3 RoPE."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"2D RoPE requires a dimension divisible by 4, got {dim}.")
        self.dim = dim
        self.base = base

    def _rotations(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        frequency_count = self.dim // 4
        frequencies = 1.0 / (
            self.base
            ** (
                torch.arange(frequency_count, device=device, dtype=torch.float32)
                / frequency_count
            )
        )
        coordinates = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        angles = torch.cat(
            [coordinate.unsqueeze(-1) * frequencies for coordinate in coordinates],
            dim=-1,
        )
        return torch.polar(torch.ones_like(angles), angles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected [B, H, W, {self.dim}] input for 2D RoPE, got {tuple(x.shape)}."
            )
        input_dtype = x.dtype
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        rotated = self._rotations(x.shape[1], x.shape[2], x.device) * x_complex
        return torch.view_as_real(rotated).flatten(-2).to(dtype=input_dtype)


class PeriodicRotaryEmbedding2D(nn.Module):
    """Two-dimensional RoPE using representations of the cyclic grid axes."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"2D RoPE requires a dimension divisible by 4, got {dim}.")
        self.dim = dim

    @staticmethod
    def _axis_angles(length: int, count: int, device: torch.device) -> torch.Tensor:
        if length <= 0:
            raise ValueError(f"Periodic axis length must be positive, got {length}.")
        coordinates = torch.arange(length, device=device, dtype=torch.float32)
        # Integer harmonics are the complex irreducible representations of Z_length.
        nontrivial_harmonics = max(length - 1, 1)
        harmonics = (
            torch.arange(count, device=device, dtype=torch.float32)
            .remainder(nontrivial_harmonics)
            .add(1.0)
        )
        return coordinates[:, None] * harmonics[None, :] * (2.0 * torch.pi / length)

    def _rotations(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        frequency_count = self.dim // 4
        height_angles = self._axis_angles(height, frequency_count, device)
        width_angles = self._axis_angles(width, frequency_count, device)
        angles = torch.cat(
            [
                height_angles[:, None, :].expand(height, width, frequency_count),
                width_angles[None, :, :].expand(height, width, frequency_count),
            ],
            dim=-1,
        )
        return torch.polar(torch.ones_like(angles), angles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected [B, H, W, {self.dim}] input for periodic 2D RoPE, "
                f"got {tuple(x.shape)}."
            )
        input_dtype = x.dtype
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        rotated = self._rotations(x.shape[1], x.shape[2], x.device) * x_complex
        return torch.view_as_real(rotated).flatten(-2).to(dtype=input_dtype)


class ConvEnhancedMlp(nn.Module):
    """H-ViT^3 MLP adapted to use whole-domain PDE boundary conditions."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwc = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            groups=hidden_features,
        )
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def reset_official_parameters(self) -> None:
        trunc_normal_(self.fc1.weight, std=0.02)
        trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        self.dwc.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
        periodic: bool,
    ) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        spatial = x.reshape(x.shape[0], height, width, x.shape[-1]).permute(0, 3, 1, 2)
        spatial = F.pad(
            spatial,
            (1, 1, 1, 1),
            mode="circular" if periodic else "constant",
        )
        x = x + self.dwc(spatial).flatten(2).transpose(1, 2)
        return self.drop(self.fc2(self.act(x)))


class GlobalViTTTMixer(nn.Module):
    """Official ViT^3 TTT mixer applied once to a complete PDE feature map."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        inner_lr: float = 1.0,
        rope_type: str = "none",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inner_lr = inner_lr
        if rope_type not in {"none", "standard", "periodic"}:
            raise ValueError(
                f"Unsupported rope_type={rope_type!r}; expected none, standard, or periodic."
            )
        self.rope_type = rope_type
        self.use_rope = rope_type != "none"

        self.qkv = nn.Linear(dim, dim * 3 + self.head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(
            torch.zeros(1, self.num_heads, self.head_dim, self.head_dim)
        )
        self.w2 = nn.Parameter(
            torch.zeros(1, self.num_heads, self.head_dim, self.head_dim)
        )
        self.w3 = nn.Parameter(torch.zeros(self.head_dim, 1, 3, 3))
        trunc_normal_(self.w1, std=0.02)
        trunc_normal_(self.w2, std=0.02)
        trunc_normal_(self.w3, std=0.02)

        self.proj = nn.Linear(dim + self.head_dim, dim)
        self.scale = 9**-0.5
        if rope_type == "standard":
            self.rope = RotaryEmbedding2D(dim)
        elif rope_type == "periodic":
            self.rope = PeriodicRotaryEmbedding2D(dim)
        else:
            self.rope = None

    def reset_official_projection_parameters(self) -> None:
        trunc_normal_(self.qkv.weight, std=0.02)
        trunc_normal_(self.proj.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def inner_train_simplified_swiglu(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = k @ w1
        z2 = k @ w2
        sigmoid = torch.sigmoid(z2)
        activation = z2 * sigmoid

        error = -v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (error * activation)
        g2 = k.transpose(-2, -1) @ (
            error * z1 * (sigmoid * (1.0 + z2 * (1.0 - sigmoid)))
        )

        # ViT^3 clips each output column independently.
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        return w1 - self.inner_lr * g1, w2 - self.inner_lr * g2

    @staticmethod
    def _pad_spatial(x: torch.Tensor, periodic: bool) -> torch.Tensor:
        return F.pad(x, (1, 1, 1, 1), mode="circular" if periodic else "constant")

    def inner_train_3x3dwc(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        periodic: bool,
    ) -> torch.Tensor:
        batch, channels, height, width = k.shape
        error = -v / float(height * width) * self.scale
        padded_k = self._pad_spatial(k, periodic)

        products = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y_start = 1 + dy
                x_start = 1 + dx
                products.append(
                    (
                        padded_k[
                            :, :, y_start : y_start + height, x_start : x_start + width
                        ]
                        * error
                    ).sum(dim=(-2, -1))
                )
        gradient = torch.stack(products, dim=-1).reshape(
            batch * channels, 1, 3, 3
        )
        gradient = gradient / (
            gradient.norm(dim=(-2, -1), keepdim=True) + 1.0
        )
        return w.repeat(batch, 1, 1, 1) - self.inner_lr * gradient

    def _apply_depthwise_fast_weights(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        periodic: bool,
    ) -> torch.Tensor:
        batch, channels, height, width = q.shape
        q = q.reshape(1, batch * channels, height, width)
        if periodic:
            q = F.pad(q, (1, 1, 1, 1), mode="circular")
            output = F.conv2d(q, weights, groups=batch * channels)
        else:
            output = F.conv2d(q, weights, padding=1, groups=batch * channels)
        return output.reshape(batch, channels, height * width).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
        periodic: bool = False,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        if channels != self.dim or tokens != height * width:
            raise ValueError(
                f"Expected [B, {height * width}, {self.dim}], got {tuple(x.shape)}."
            )

        head_dim = self.head_dim
        q1, k1, v1, q2, k2, v2 = torch.split(
            self.qkv(x),
            [channels, channels, channels, head_dim, head_dim, head_dim],
            dim=-1,
        )

        if self.rope is not None:
            q1 = self.rope(q1.reshape(batch, height, width, channels))
            k1 = self.rope(k1.reshape(batch, height, width, channels))

        q1 = q1.reshape(batch, tokens, self.num_heads, head_dim).transpose(1, 2)
        k1 = k1.reshape(batch, tokens, self.num_heads, head_dim).transpose(1, 2)
        v1 = v1.reshape(batch, tokens, self.num_heads, head_dim).transpose(1, 2)
        q2 = q2.reshape(batch, height, width, head_dim).permute(0, 3, 1, 2)
        k2 = k2.reshape(batch, height, width, head_dim).permute(0, 3, 1, 2)
        v2 = v2.reshape(batch, height, width, head_dim).permute(0, 3, 1, 2)

        # These are per-sample fast weights. The learned initial weights are never mutated.
        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3x3dwc(k2, v2, self.w3, periodic=periodic)

        output1 = (q1 @ w1) * F.silu(q1 @ w2)
        output1 = output1.transpose(1, 2).reshape(batch, tokens, channels)
        output2 = self._apply_depthwise_fast_weights(q2, w3, periodic=periodic)
        return self.proj(torch.cat([output1, output2], dim=-1))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"inner_lr={self.inner_lr}, rope_type={self.rope_type}"
        )
