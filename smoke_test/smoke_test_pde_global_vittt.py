"""Numerical and integration checks for full-map ViTTT PDE mixers."""

import importlib
import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TTT = (
    REPO_ROOT.parent / "_external_research" / "ViTTT-latest" / "vittt" / "models" / "ttt_block.py"
)
UPSTREAM_REPO = REPO_ROOT.parent / "pde-transformer"
CONFIG_PATHS = {
    "attention": REPO_ROOT / "server_example" / "pdes_attention_128_60sims.yaml",
    "global_vittt": REPO_ROOT
    / "server_example"
    / "pdes_global-vittt_128_60sims.yaml",
    "global_h_vittt": REPO_ROOT
    / "server_example"
    / "pdes_global-h-vittt_128_60sims.yaml",
}


def load_experiment_modules():
    """Load only mixed_channels, avoiding unrelated optional model imports."""
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
    global_vittt = importlib.import_module("pdetransformer.core.pde_vittt_global")
    return pde, global_vittt


def load_upstream_pde_module():
    if not UPSTREAM_REPO.is_dir():
        raise FileNotFoundError(f"Upstream PDE source was not found at {UPSTREAM_REPO}")
    package_paths = {
        "upstream_pdetransformer": UPSTREAM_REPO / "pdetransformer",
        "upstream_pdetransformer.core": UPSTREAM_REPO / "pdetransformer" / "core",
        "upstream_pdetransformer.core.mixed_channels": (
            UPSTREAM_REPO / "pdetransformer" / "core" / "mixed_channels"
        ),
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return importlib.import_module(
        "upstream_pdetransformer.core.mixed_channels.pde_transformer"
    )


pde_module, global_vittt_module = load_experiment_modules()
PDETransformer = pde_module.PDETransformer
DepthwiseCPE2D = global_vittt_module.DepthwiseCPE2D
GlobalViTTTMixer = global_vittt_module.GlobalViTTTMixer


def assert_attention_regression_parity():
    upstream = load_upstream_pde_module()
    model_args = {
        "sample_size": 32,
        "in_channels": 2,
        "out_channels": 2,
        "type": "PDE-S",
        "periodic": True,
        "carrier_token_active": False,
    }
    torch.manual_seed(5)
    reference = upstream.PDETransformer(**model_args).eval()
    torch.manual_seed(5)
    implementation = PDETransformer(**model_args, token_mixer_type="attention").eval()

    reference_state = reference.state_dict()
    implementation_state = implementation.state_dict()
    assert reference_state.keys() == implementation_state.keys()
    for name in reference_state:
        torch.testing.assert_close(
            implementation_state[name], reference_state[name], atol=0.0, rtol=0.0
        )

    x = torch.randn(1, 2, 32, 32)
    t = torch.zeros(1)
    y = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        expected = reference(x, t, y).sample
        actual = implementation(x, t, y).sample
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def assert_fair_training_configs():
    loaded = {
        mixer: OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        for mixer, path in CONFIG_PATHS.items()
    }
    for mixer, config in loaded.items():
        assert config["token_mixer_type"] == mixer

    comparable = []
    for config in loaded.values():
        config = dict(config)
        config.pop("run_name")
        config.pop("token_mixer_type")
        comparable.append(config)
    assert comparable[0] == comparable[1] == comparable[2]


def load_official_ttt():
    if not OFFICIAL_TTT.is_file():
        raise FileNotFoundError(f"Official ViTTT source was not found at {OFFICIAL_TTT}")
    spec = importlib.util.spec_from_file_location("official_vittt_ttt_block", OFFICIAL_TTT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TTT


def assert_official_zero_padding_parity():
    official_ttt = load_official_ttt()
    torch.manual_seed(7)
    reference = official_ttt(dim=16, num_heads=4, qkv_bias=True)
    implementation = GlobalViTTTMixer(dim=16, num_heads=4, qkv_bias=True)
    implementation.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 16, 16)

    with torch.no_grad():
        expected = reference(x, 4, 4)
        actual = implementation(x, height=4, width=4, periodic=False)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def assert_fast_weights_reset_per_forward():
    torch.manual_seed(11)
    mixer = GlobalViTTTMixer(dim=16, num_heads=4)
    x = torch.randn(2, 16, 16)
    parameters_before = {name: value.detach().clone() for name, value in mixer.named_parameters()}

    with torch.no_grad():
        first = mixer(x, height=4, width=4, periodic=True)
        second = mixer(x, height=4, width=4, periodic=True)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    for name, value in mixer.named_parameters():
        torch.testing.assert_close(value, parameters_before[name], atol=0.0, rtol=0.0)

    other_batch = x.clone()
    other_batch[1] = torch.randn_like(other_batch[1])
    with torch.no_grad():
        changed = mixer(other_batch, height=4, width=4, periodic=True)
    torch.testing.assert_close(first[0], changed[0], atol=1e-6, rtol=1e-6)


def assert_global_periodic_boundary():
    cpe = DepthwiseCPE2D(dim=1)
    with torch.no_grad():
        cpe.proj.weight.zero_()
        cpe.proj.bias.zero_()
        cpe.proj.weight[0, 0, 0, 1] = 1.0

    x = torch.zeros(1, 1, 4, 4)
    x[0, 0, 3, 0] = 2.0
    with torch.no_grad():
        periodic = cpe(x, periodic=True)
        zero_padded = cpe(x, periodic=False)
    assert periodic[0, 0, 0, 0].item() == 2.0
    assert zero_padded[0, 0, 0, 0].item() == 0.0


def assert_full_model_uses_global_tokens(token_mixer_type: str):
    torch.manual_seed(13)
    model = PDETransformer(
        sample_size=128,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        periodic=True,
        carrier_token_active=False,
        token_mixer_type=token_mixer_type,
    ).eval()

    mixers = [module for module in model.modules() if isinstance(module, GlobalViTTTMixer)]
    cpes = [module for module in model.modules() if isinstance(module, DepthwiseCPE2D)]
    assert len(mixers) == 22
    assert len(cpes) == 22
    assert len({id(module.w1) for module in mixers}) == 22
    assert len({id(module.proj.weight) for module in cpes}) == 22
    assert all(module.use_rope == (token_mixer_type == "global_h_vittt") for module in mixers)

    calls = []
    original_forward = GlobalViTTTMixer.forward

    def recording_forward(self, x, height, width, periodic=False):
        calls.append((x.shape[1], x.shape[2], height, width))
        return original_forward(self, x, height, width, periodic)

    def forbidden(*args, **kwargs):
        raise AssertionError("Global ViTTT must not use window partitioning or shifted windows.")

    x = torch.randn(1, 2, 128, 128)
    t = torch.zeros(1)
    y = torch.zeros(1, dtype=torch.long)
    with (
        patch.object(GlobalViTTTMixer, "forward", recording_forward),
        patch.object(pde_module, "window_partition", forbidden),
        patch.object(pde_module, "window_reverse", forbidden),
        patch.object(pde_module.torch, "roll", forbidden),
        torch.no_grad(),
    ):
        output = model(x, timestep=t, class_labels=y).sample

    assert output.shape == x.shape
    observed = Counter((tokens, channels) for tokens, channels, _, _ in calls)
    expected = Counter(
        {
            (1024, 96): 2,
            (256, 192): 7,
            (64, 384): 8,
            (1024, 192): 5,
        }
    )
    assert observed == expected, f"Unexpected full-map calls: {observed}"
    assert all(tokens == height * width for tokens, _, height, width in calls)


def main():
    checks = [
        ("upstream Attention regression parity", assert_attention_regression_parity),
        ("official zero-padding parity", assert_official_zero_padding_parity),
        ("per-forward fast-weight reset", assert_fast_weights_reset_per_forward),
        ("global periodic boundary", assert_global_periodic_boundary),
        ("fair training configurations", assert_fair_training_configs),
        (
            "ViTTT-PDE full-model integration",
            lambda: assert_full_model_uses_global_tokens("global_vittt"),
        ),
        (
            "H-ViTTT-PDE full-model integration",
            lambda: assert_full_model_uses_global_tokens("global_h_vittt"),
        ),
    ]
    for name, check in checks:
        check()
        print(f"PASS: {name}")


if __name__ == "__main__":
    main()
