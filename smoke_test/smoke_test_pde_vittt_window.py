from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

module_path = REPO_ROOT / "pdetransformer" / "core" / "pde_vittt_window.py"
spec = importlib.util.spec_from_file_location("pde_vittt_window", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {module_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PDEViTTTWindowBlock = module.PDEViTTTWindowBlock


def run_case(periodic: bool, padding_mode: str) -> None:
    torch.manual_seed(0)
    block = PDEViTTTWindowBlock(
        dim=32,
        num_heads=4,
        inner_lr=1.0,
        padding_mode=padding_mode,
    )
    x = torch.randn(3, 16, 32, requires_grad=True)
    y = block(x, h=4, w=4, periodic=periodic)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    loss = y.square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    for name, param in block.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def main() -> None:
    run_case(periodic=True, padding_mode="zero")
    run_case(periodic=False, padding_mode="zero")
    run_case(periodic=False, padding_mode="replicate")
    print("PDEViTTTWindowBlock smoke test passed")


if __name__ == "__main__":
    main()
