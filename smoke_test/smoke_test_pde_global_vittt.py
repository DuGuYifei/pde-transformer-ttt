"""Numerical and integration checks for full-map ViTTT PDE mixers."""

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
OFFICIAL_TTT = (
    REPO_ROOT.parent / "_external_research" / "ViTTT-latest" / "vittt" / "models" / "ttt_block.py"
)
OFFICIAL_VITTT_ROOT = REPO_ROOT.parent / "_external_research" / "ViTTT-latest" / "vittt"
UPSTREAM_REPO = REPO_ROOT.parent / "pde-transformer"
CONFIG_PATHS = {
    "attention": REPO_ROOT / "server_example" / "pdes_attention_128_100ep_60sims.yaml",
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
ConvEnhancedMlp = global_vittt_module.ConvEnhancedMlp
PeriodicRotaryEmbedding2D = global_vittt_module.PeriodicRotaryEmbedding2D


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


def assert_pretrained_round_trip():
    model_type = "PDE-TINY-VITTT-SAVE-LOAD"

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
            token_mixer_type="global_h_vittt",
            vittt_inner_lr=0.375,
            vittt_head_dim=16,
        ).eval()
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            loaded = PDETransformer.from_pretrained(directory).eval()

        expected_config = {
            "token_mixer_type": "global_h_vittt",
            "vittt_inner_lr": 0.375,
            "vittt_head_dim": 16,
        }
        for name, value in expected_config.items():
            assert getattr(loaded.config, name) == value

        loaded_mixers = [
            module for module in loaded.modules() if isinstance(module, GlobalViTTTMixer)
        ]
        assert loaded_mixers
        assert all(module.head_dim == 16 for module in loaded_mixers)
        assert all(module.inner_lr == 0.375 for module in loaded_mixers)
        assert all(module.rope_type == "periodic" for module in loaded_mixers)

        expected_state = model.state_dict()
        loaded_state = loaded.state_dict()
        assert expected_state.keys() == loaded_state.keys()
        for name in expected_state:
            torch.testing.assert_close(
                loaded_state[name], expected_state[name], atol=0.0, rtol=0.0
            )
    finally:
        del pde_module.PDE_models[model_type]


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
        # Duration is checked by each experiment-specific test.
        config.pop("max_epochs")
        comparable.append(config)
    assert comparable
    assert all(config == comparable[0] for config in comparable[1:])


def load_official_ttt():
    if not OFFICIAL_TTT.is_file():
        raise FileNotFoundError(f"Official ViTTT source was not found at {OFFICIAL_TTT}")
    spec = importlib.util.spec_from_file_location("official_vittt_ttt_block", OFFICIAL_TTT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TTT


def load_official_h_vittt():
    package_paths = {
        "official_vittt": OFFICIAL_VITTT_ROOT,
        "official_vittt.models": OFFICIAL_VITTT_ROOT / "models",
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return importlib.import_module("official_vittt.models.h_vittt")


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


def assert_periodic_rope_boundary_phase():
    rope = PeriodicRotaryEmbedding2D(dim=16)
    rotations = rope._rotations(height=4, width=5, device=torch.device("cpu"))
    frequency_count = rope.dim // 4

    height_rotations = rotations[:, :, :frequency_count]
    height_seam_step = height_rotations[0, 0] * height_rotations[-1, 0].conj()
    height_regular_step = height_rotations[1, 0] * height_rotations[0, 0].conj()
    torch.testing.assert_close(height_seam_step, height_regular_step)

    width_rotations = rotations[:, :, frequency_count:]
    width_seam_step = width_rotations[0, 0] * width_rotations[0, -1].conj()
    width_regular_step = width_rotations[0, 1] * width_rotations[0, 0].conj()
    torch.testing.assert_close(width_seam_step, width_regular_step)


def assert_rope_rotation_cache():
    rope = PeriodicRotaryEmbedding2D(dim=16)
    x = torch.randn(2, 4, 5, 16)
    with patch.object(rope, "_rotations", wraps=rope._rotations) as rotations:
        expected = rope(x)
        actual = rope(x.clone())
        assert rotations.call_count == 1

        rope(torch.randn(1, 3, 5, 16))
        assert rotations.call_count == 2

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert len(rope._rotation_cache) == 2
    rope.to(torch.device("cpu"))
    assert not rope._rotation_cache


def assert_h_mlp_official_parity():
    official_h_vittt = load_official_h_vittt()
    torch.manual_seed(17)
    reference = official_h_vittt.Mlp(in_features=8, hidden_features=16)
    implementation = ConvEnhancedMlp(in_features=8, hidden_features=16)
    implementation.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 16, 8)

    with torch.no_grad():
        expected = reference(x, 4, 4)
        actual = implementation(x, height=4, width=4, periodic=False)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def assert_h_mlp_periodic_boundary():
    mlp = ConvEnhancedMlp(
        in_features=1,
        hidden_features=1,
        act_layer=torch.nn.Identity,
    )
    with torch.no_grad():
        mlp.fc1.weight.fill_(1.0)
        mlp.fc1.bias.zero_()
        mlp.fc2.weight.fill_(1.0)
        mlp.fc2.bias.zero_()
        mlp.dwc.weight.zero_()
        mlp.dwc.bias.zero_()
        mlp.dwc.weight[0, 0, 0, 1] = 1.0

    x = torch.zeros(1, 16, 1)
    x[0, 12, 0] = 2.0
    with torch.no_grad():
        periodic = mlp(x, height=4, width=4, periodic=True)
        zero_padded = mlp(x, height=4, width=4, periodic=False)
    assert periodic[0, 0, 0].item() == 2.0
    assert zero_padded[0, 0, 0].item() == 0.0


def assert_rope_true_half_precision():
    for dtype in (torch.float16, torch.bfloat16):
        mixer = GlobalViTTTMixer(
            dim=16,
            num_heads=4,
            rope_type="periodic",
        ).to(dtype=dtype)
        x = torch.randn(2, 16, 16, dtype=dtype)
        with torch.no_grad():
            output = mixer(x, height=4, width=4, periodic=True)
        assert output.dtype == dtype
        assert torch.isfinite(output).all()


def assert_h_components_backward():
    torch.manual_seed(19)
    mixer = GlobalViTTTMixer(
        dim=16,
        num_heads=4,
        rope_type="periodic",
    )
    mlp = ConvEnhancedMlp(in_features=16, hidden_features=32)
    x = torch.randn(2, 16, 16, requires_grad=True)
    output = mixer(x, height=4, width=4, periodic=True)
    output = mlp(output, height=4, width=4, periodic=True)
    output.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    parameters = list(mixer.parameters()) + list(mlp.parameters())
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


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
    conv_mlps = [module for module in model.modules() if isinstance(module, ConvEnhancedMlp)]
    assert len(mixers) == 22
    assert len(cpes) == 22
    assert len(conv_mlps) == (22 if token_mixer_type == "global_h_vittt" else 0)
    assert len({id(module.w1) for module in mixers}) == 22
    assert len({id(module.proj.weight) for module in cpes}) == 22
    assert all(module.head_dim == 32 for module in mixers)
    expected_heads = {96: 3, 192: 6, 384: 12}
    assert all(module.num_heads == expected_heads[module.dim] for module in mixers)
    expected_rope = "periodic" if token_mixer_type == "global_h_vittt" else "none"
    assert all(module.rope_type == expected_rope for module in mixers)

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
        ("Diffusers save/load round trip", assert_pretrained_round_trip),
        ("official zero-padding parity", assert_official_zero_padding_parity),
        ("per-forward fast-weight reset", assert_fast_weights_reset_per_forward),
        ("periodic RoPE boundary phase", assert_periodic_rope_boundary_phase),
        ("RoPE rotation cache", assert_rope_rotation_cache),
        ("H-style MLP official parity", assert_h_mlp_official_parity),
        ("H-style MLP periodic boundary", assert_h_mlp_periodic_boundary),
        ("RoPE true half precision", assert_rope_true_half_precision),
        ("H-style component backward", assert_h_components_backward),
        ("global periodic boundary", assert_global_periodic_boundary),
        ("fair training configurations", assert_fair_training_configs),
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
