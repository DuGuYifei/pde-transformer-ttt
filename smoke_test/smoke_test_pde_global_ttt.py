from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pdetransformer.core.pde_global_ttt import PDEGlobalViTTT2D
from pdetransformer.core.mixed_channels.pde_transformer import PDETransformer


def test_global_module() -> None:
    torch.manual_seed(0)
    module = PDEGlobalViTTT2D(dim=32, num_heads=4, gate_init=0.0)
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    y = module(x, periodic=True)
    assert torch.equal(y, x)
    y.square().mean().backward()
    assert module.gate.grad is not None
    assert torch.isfinite(module.gate.grad).all()


def test_full_model() -> None:
    torch.manual_seed(0)
    base = PDETransformer(
        sample_size=64,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        patch_size=4,
        periodic=True,
        carrier_token_active=False,
        token_mixer_type="attention",
    )
    model = PDETransformer(
        sample_size=64,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        patch_size=4,
        periodic=True,
        carrier_token_active=False,
        token_mixer_type="attention",
        global_ttt_stage_names=["encoder_0"],
    )
    assert model.model.encoder_level_0.global_ttt is not None
    assert model.model.encoder_level_1.global_ttt is None
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(".global_ttt." in key for key in missing)
    assert not unexpected
    x = torch.randn(1, 2, 64, 64)
    labels = torch.zeros(1, dtype=torch.long)
    base.eval()
    model.eval()
    with torch.no_grad():
        baseline_y = base(x, class_labels=labels).sample
        y = model(x, class_labels=labels).sample
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.equal(y, baseline_y)


if __name__ == "__main__":
    test_global_module()
    test_full_model()
    print("PDE global ViTTT smoke test passed")
