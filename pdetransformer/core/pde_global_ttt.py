from __future__ import annotations

import torch
from torch import nn

from .pde_vittt_window import PDEViTTTWindowBlock


class PDEGlobalViTTT2D(nn.Module):
    """Global linear-complexity ViTTT update over a complete PDE feature map."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        inner_lr: float = 1.0,
        gate_init: float = 0.0,
        padding_mode: str = "zero",
        key_instance_norm: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mixer = PDEViTTTWindowBlock(
            dim=dim,
            num_heads=num_heads,
            inner_lr=inner_lr,
            padding_mode=padding_mode,
            key_instance_norm=key_instance_norm,
        )
        self.gate = nn.Parameter(torch.full((dim,), float(gate_init)))

    def forward(self, x: torch.Tensor, periodic: bool = False) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"expected {self.dim} channels, got {channels}")

        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        update = self.mixer(self.norm(tokens), h=height, w=width, periodic=periodic)
        update = update.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        gate = torch.tanh(self.gate).view(1, channels, 1, 1)
        return x + gate * update
