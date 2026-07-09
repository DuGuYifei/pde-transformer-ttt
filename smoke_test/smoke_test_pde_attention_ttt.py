from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pdetransformer.core.mixed_channels.pde_transformer import PDEBlock  # noqa: E402


def run_case(ttt_layer_type: str, bidirectional: bool) -> None:
    torch.manual_seed(0)
    block = PDEBlock(
        dim=32,
        num_heads=4,
        window_size=4,
        periodic=True,
        carrier_token_active=False,
        token_mixer_type="attention_ttt",
        ttt_layer_type=ttt_layer_type,
        ttt_mini_batch_size=16,
        ttt_base_lr=1.0,
        attention_ttt_type="ttt_sequence",
        attention_ttt_gate_init=0.1,
        attention_ttt_bidirectional=bidirectional,
    )
    x = torch.randn(2, 4, 4, 32, requires_grad=True)
    emb = torch.zeros(2, 32)
    y, _ = block(x, carrier_tokens=None, emb=emb)
    assert y.shape == (2, 16, 32)
    assert torch.isfinite(y).all()

    loss = y.square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    for name, param in block.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def main() -> None:
    run_case(ttt_layer_type="linear", bidirectional=False)
    run_case(ttt_layer_type="mlp", bidirectional=True)
    print("PDE attention_ttt smoke test passed")


if __name__ == "__main__":
    main()
