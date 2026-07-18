from __future__ import annotations

import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name, package_path in {
    "pdetransformer": ROOT / "pdetransformer",
    "pdetransformer.core": ROOT / "pdetransformer" / "core",
}.items():
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

from pdetransformer.core.mixed_channels import PDETransformer  # noqa: E402
from pdetransformer.core.mixed_channels.train_rollout import (  # noqa: E402
    PersistentAutoregressiveRolloutSupervised,
    detach_state_tree,
)
from pdetransformer.core.pde_vittt_global_linear import (  # noqa: E402
    GlobalLinearTTTMixer,
)
from server_example.train_global_linear_rollout_server import (  # noqa: E402
    build_training_module,
    load_config,
)


CONFIG = (
    ROOT
    / "server_example"
    / "pdes_global-linear-r29-persistent_128_100ep_60sims.yaml"
)


def flatten_state(state: dict) -> list[torch.Tensor]:
    return [tensor for stage in state.values() for tensor in stage]


def test_mixer_recurrence_and_temporal_gradient() -> None:
    torch.manual_seed(7)
    mixer = GlobalLinearTTTMixer(dim=32, num_heads=1)
    x1 = torch.randn(1, 4, 32, requires_grad=True)
    x2 = torch.randn(1, 4, 32, requires_grad=True)
    _, state1 = mixer(
        x1, height=2, width=2, return_fast_weights=True
    )
    output2, state2 = mixer(
        x2,
        height=2,
        width=2,
        fast_weights=state1,
        return_fast_weights=True,
    )
    _, reset_state2 = mixer(
        x2, height=2, width=2, return_fast_weights=True
    )
    assert not torch.equal(state2, reset_state2)
    output2.square().mean().backward()
    assert x1.grad is not None and x1.grad.abs().sum().item() > 0
    assert x2.grad is not None and x2.grad.abs().sum().item() > 0


def test_full_model_state_lifecycle() -> None:
    torch.manual_seed(11)
    model = PDETransformer(
        sample_size=32,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        patch_size=4,
        periodic=True,
        carrier_token_active=False,
        token_mixer_type="global_linear_ttt",
        vittt_inner_lr=1.0,
        vittt_head_dim=32,
        vittt_persistent_state=True,
    ).eval()
    x = torch.randn(1, 2, 32, 32)
    labels = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        first = model(x, class_labels=labels, return_ttt_state_cache=True)
        continued = model(
            x,
            class_labels=labels,
            ttt_state_cache=first.ttt_state_cache,
            return_ttt_state_cache=True,
        )
        reset = model(x, class_labels=labels, return_ttt_state_cache=True)

    expected_depths = {
        "encoder_level_0": 2,
        "encoder_level_1": 5,
        "latent": 8,
        "decoder_level_1": 2,
        "decoder_level_0": 5,
    }
    assert {key: len(value) for key, value in first.ttt_state_cache.items()} == expected_depths
    assert len(flatten_state(first.ttt_state_cache)) == 22
    for first_state, reset_state in zip(
        flatten_state(first.ttt_state_cache), flatten_state(reset.ttt_state_cache)
    ):
        torch.testing.assert_close(first_state, reset_state)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            flatten_state(continued.ttt_state_cache),
            flatten_state(reset.ttt_state_cache),
        )
    )
    detached = detach_state_tree(continued.ttt_state_cache)
    assert all(not tensor.requires_grad for tensor in flatten_state(detached))


def test_config_checkpoint_compatibility_and_save_load() -> None:
    config = load_config(CONFIG)
    assert config["vittt_persistent_state"] is True
    assert config["train_unrolling_steps"] == 29
    assert config["tbptt_chunk_size"] == 4
    module = build_training_module(config)
    assert isinstance(module, PersistentAutoregressiveRolloutSupervised)

    stateless = PDETransformer(
        sample_size=32,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        patch_size=4,
        periodic=True,
        carrier_token_active=False,
        token_mixer_type="global_linear_ttt",
        vittt_persistent_state=False,
    )
    persistent = PDETransformer(
        sample_size=32,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        patch_size=4,
        periodic=True,
        carrier_token_active=False,
        token_mixer_type="global_linear_ttt",
        vittt_persistent_state=True,
    )
    persistent.load_state_dict(stateless.state_dict(), strict=True)

    with TemporaryDirectory() as directory:
        persistent.save_pretrained(directory)
        restored = PDETransformer.from_pretrained(directory)
        assert restored.config.vittt_persistent_state is True
        restored.load_state_dict(persistent.state_dict(), strict=True)


def test_invalid_state_usage_is_rejected() -> None:
    model = PDETransformer(
        sample_size=32,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        patch_size=4,
        token_mixer_type="global_linear_ttt",
        vittt_persistent_state=False,
    )
    x = torch.randn(1, 2, 32, 32)
    try:
        model(x, return_ttt_state_cache=True)
    except ValueError as error:
        assert "vittt_persistent_state" in str(error)
    else:
        raise AssertionError("A stateless model must reject persistent-state output")


def test_cuda_low_precision_state() -> None:
    if not torch.cuda.is_available():
        return
    for dtype in (torch.float16, torch.bfloat16):
        mixer = GlobalLinearTTTMixer(dim=32, num_heads=1).cuda().to(dtype=dtype)
        x1 = torch.randn(1, 4, 32, device="cuda", dtype=dtype, requires_grad=True)
        x2 = torch.randn(1, 4, 32, device="cuda", dtype=dtype, requires_grad=True)
        _, state = mixer(x1, height=2, width=2, return_fast_weights=True)
        output, state = mixer(
            x2,
            height=2,
            width=2,
            fast_weights=state,
            return_fast_weights=True,
        )
        output.float().square().mean().backward()
        assert state.dtype == dtype
        assert x1.grad is not None and x2.grad is not None


if __name__ == "__main__":
    test_mixer_recurrence_and_temporal_gradient()
    print("PASS: mixer recurrence and cross-step gradient")
    test_full_model_state_lifecycle()
    print("PASS: 22-block explicit state lifecycle")
    test_config_checkpoint_compatibility_and_save_load()
    print("PASS: config, checkpoint compatibility, and save/load")
    test_invalid_state_usage_is_rejected()
    print("PASS: invalid state guard")
    test_cuda_low_precision_state()
    print("PASS: CUDA persistent low precision (when available)")
