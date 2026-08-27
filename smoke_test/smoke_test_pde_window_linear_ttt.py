"""Checks for closed-form linear TTT applied independently in shifted windows."""

import importlib
import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOW_CONFIG = (
    REPO_ROOT / "server_example" / "pdes_window-linear-ttt_128_60sims.yaml"
)
ATTENTION_CONFIG = (
    REPO_ROOT / "server_example" / "pdes_attention_128_100ep_60sims.yaml"
)


def load_experiment_modules():
    package_paths = {
        "pdetransformer": REPO_ROOT / "pdetransformer",
        "pdetransformer.core": REPO_ROOT / "pdetransformer" / "core",
        "pdetransformer.core.mixed_channels": (
            REPO_ROOT / "pdetransformer" / "core" / "mixed_channels"
        ),
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    pde = importlib.import_module(
        "pdetransformer.core.mixed_channels.pde_transformer"
    )
    linear = importlib.import_module("pdetransformer.core.pde_vittt_global_linear")
    return pde, linear


pde_module, linear_module = load_experiment_modules()
PDETransformer = pde_module.PDETransformer
GlobalLinearTTTMixer = linear_module.GlobalLinearTTTMixer
WindowAttention2DTime = pde_module.WindowAttention2DTime


def explicit_reference(mixer, x):
    batch, token_count, _ = x.shape
    qkv = mixer.qkv(x).reshape(
        batch, token_count, 3, mixer.num_heads, mixer.head_dim
    )
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    gradient = -(mixer.scale / float(token_count)) * (
        k.transpose(-2, -1) @ v
    )
    gradient = gradient / (gradient.norm(dim=-2, keepdim=True) + 1.0)
    fast_weights = mixer.w0 - mixer.inner_lr * gradient
    output = (q @ fast_weights).transpose(1, 2).reshape(
        batch, token_count, mixer.dim
    )
    return output, fast_weights


def assert_reference_formula_and_reset():
    torch.manual_seed(3)
    mixer = GlobalLinearTTTMixer(dim=32, num_heads=1, inner_lr=0.75).eval()
    windows = torch.randn(3, 64, 32)
    w0_before = mixer.w0.detach().clone()
    with torch.no_grad():
        expected, fast_weights = explicit_reference(mixer, windows)
        actual = mixer(windows, height=8, width=8)
        repeated = mixer(windows, height=8, width=8)

    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual, repeated, atol=0.0, rtol=0.0)
    torch.testing.assert_close(mixer.w0, w0_before, atol=0.0, rtol=0.0)
    assert fast_weights.shape == (3, 1, 32, 32)
    assert not torch.allclose(fast_weights[0], fast_weights[1])


def build_model():
    return PDETransformer(
        sample_size=128,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        periodic=True,
        carrier_token_active=False,
        token_mixer_type="window_linear_ttt",
        vittt_inner_lr=1.0,
        vittt_head_dim=32,
    ).eval()


def assert_window_model_integration():
    torch.manual_seed(7)
    model = build_model()
    mixers = [
        module for module in model.modules()
        if isinstance(module, GlobalLinearTTTMixer)
    ]
    assert len(mixers) == 22
    assert len({id(module.w0) for module in mixers}) == 22
    assert not any(type(module).__name__ == "DepthwiseCPE2D" for module in model.modules())
    assert not any(
        isinstance(module, WindowAttention2DTime) for module in model.modules()
    )
    assert all(module.head_dim == 32 for module in mixers)

    calls = []
    rolls = []
    original_mixer_forward = GlobalLinearTTTMixer.forward
    original_roll = pde_module.torch.roll

    def recording_forward(
        self, x, height, width, periodic=False, windows_per_sample=1
    ):
        calls.append((x.shape[0], x.shape[1], x.shape[2], height, width))
        return original_mixer_forward(
            self,
            x,
            height,
            width,
            periodic,
            windows_per_sample=windows_per_sample,
        )

    def recording_roll(*args, **kwargs):
        shifts = kwargs.get("shifts", args[1] if len(args) > 1 else None)
        rolls.append(shifts)
        return original_roll(*args, **kwargs)

    x = torch.randn(1, 2, 128, 128)
    with (
        patch.object(GlobalLinearTTTMixer, "forward", recording_forward),
        patch.object(pde_module.torch, "roll", recording_roll),
        torch.no_grad(),
    ):
        output = model(
            x,
            timestep=torch.zeros(1),
            class_labels=torch.zeros(1, dtype=torch.long),
        ).sample

    assert output.shape == x.shape
    assert len(calls) == 22
    assert all(tokens == 64 and height == 8 and width == 8 for _, tokens, _, height, width in calls)
    observed = Counter((windows, channels) for windows, _, channels, _, _ in calls)
    expected = Counter({(16, 96): 2, (4, 192): 7, (1, 384): 8, (16, 192): 5})
    assert observed == expected, f"Unexpected window calls: {observed}"
    assert rolls, "Alternating shifted windows must remain active."
    assert all(abs(shift[0]) == 4 and abs(shift[1]) == 4 for shift in rolls)


def assert_backward():
    mixer = GlobalLinearTTTMixer(dim=32, num_heads=1)
    x = torch.randn(2, 64, 32, requires_grad=True)
    output = mixer(x, height=8, width=8)
    output.square().mean().backward()
    assert mixer.qkv.weight.grad is not None
    assert mixer.w0.grad is not None
    assert torch.isfinite(mixer.qkv.weight.grad).all()
    assert torch.isfinite(mixer.w0.grad).all()


def assert_pretrained_round_trip():
    model_type = "PDE-TINY-WINDOW-LINEAR-TTT-SAVE-LOAD"

    def build_tiny_model(**kwargs):
        return pde_module.PDEImpl(
            hidden_size=32,
            max_hidden_size=128,
            num_heads=4,
            depth=[1, 1, 1],
            mlp_ratio=2,
            **kwargs,
        )

    pde_module.PDE_models[model_type] = build_tiny_model
    try:
        model = PDETransformer(
            sample_size=32,
            in_channels=2,
            out_channels=2,
            type=model_type,
            periodic=True,
            carrier_token_active=False,
            token_mixer_type="window_linear_ttt",
            vittt_inner_lr=0.375,
            vittt_head_dim=16,
        ).eval()
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            loaded = PDETransformer.from_pretrained(directory).eval()

        assert loaded.config.token_mixer_type == "window_linear_ttt"
        assert loaded.config.vittt_inner_lr == 0.375
        assert loaded.config.vittt_head_dim == 16
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                loaded.state_dict()[name], value, atol=0.0, rtol=0.0
            )
    finally:
        del pde_module.PDE_models[model_type]


def assert_matched_config_and_entrypoint():
    window = OmegaConf.to_container(OmegaConf.load(WINDOW_CONFIG), resolve=True)
    attention = OmegaConf.to_container(
        OmegaConf.load(ATTENTION_CONFIG), resolve=True
    )
    assert window["token_mixer_type"] == "window_linear_ttt"
    assert attention["token_mixer_type"] == "attention"
    for config in (window, attention):
        config.pop("run_root")
        config.pop("run_name")
        config.pop("token_mixer_type")
    assert window == attention

    mixed_channels = sys.modules["pdetransformer.core.mixed_channels"]
    mixed_channels.PDETransformer = pde_module.PDETransformer
    mixed_channels.SingleStepSupervised = importlib.import_module(
        "pdetransformer.core.mixed_channels.train_supervised"
    ).SingleStepSupervised
    path = REPO_ROOT / "server_example" / "train_global_vittt_ape_xxl_server.py"
    spec = importlib.util.spec_from_file_location("window_linear_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    loaded = module.load_config(WINDOW_CONFIG)
    assert loaded["token_mixer_type"] == "window_linear_ttt"


def main():
    checks = [
        ("closed-form reference and per-call reset", assert_reference_formula_and_reset),
        ("windowed full-model integration", assert_window_model_integration),
        ("CPU FP32 backward", assert_backward),
        ("Diffusers save/load round trip", assert_pretrained_round_trip),
        ("matched config and training entrypoint", assert_matched_config_and_entrypoint),
    ]
    for name, check in checks:
        check()
        print(f"PASS: {name}")


if __name__ == "__main__":
    main()
