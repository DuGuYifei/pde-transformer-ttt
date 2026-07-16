from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .ttt_window_attention import TTTWindowAttention2DTime


class PDETemporalTTT2D(nn.Module):
    """Stateful TTT mixer over one spatial latent map per physical time step."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        layer_type: str = "mlp",
        mini_batch_size: int = 64,
        base_lr: float = 1.0,
        gate_init: float = 0.1,
        use_output_gate: bool = False,
        scan_checkpoint_group_size: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mixer = TTTWindowAttention2DTime(
            dim=dim,
            num_heads=num_heads,
            ttt_layer_type=layer_type,
            ttt_base_lr=base_lr,
            mini_batch_size=mini_batch_size,
            use_gate=use_output_gate,
            scan_checkpoint_group_size=scan_checkpoint_group_size,
        )
        self.gate = nn.Parameter(torch.full((dim,), float(gate_init)))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"expected {self.dim} channels, got {channels}")

        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        update, next_state = self.mixer(
            self.norm(tokens),
            ttt_state=state,
            return_ttt_state=True,
        )
        update = update.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        gate = torch.tanh(self.gate).view(1, channels, 1, 1)
        return x + gate * update, next_state
