"""Numerical and integration checks for the full-map linear TTT PDE mixer."""

import importlib
import importlib.util
import inspect
import sys
import types
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
LINEAR_CONFIG = (
    REPO_ROOT
    / "server_example"
    / "pdes_global-linear-ttt_128_60sims.yaml"
)
PLAIN_CONFIG = (
    REPO_ROOT
    / "server_example"
    / "pdes_attention_128_100ep_60sims.yaml"
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
    global_vittt = importlib.import_module("pdetransformer.core.pde_vittt_global")
    linear_ttt = importlib.import_module(
        "pdetransformer.core.pde_vittt_global_linear"
    )
    return pde, global_vittt, linear_ttt


pde_module, global_vittt_module, linear_ttt_module = load_experiment_modules()
PDETransformer = pde_module.PDETransformer
DepthwiseCPE2D = global_vittt_module.DepthwiseCPE2D
ConvEnhancedMlp = global_vittt_module.ConvEnhancedMlp
GlobalLinearTTTMixer = linear_ttt_module.GlobalLinearTTTMixer


def explicit_reference(mixer: GlobalLinearTTTMixer, x: torch.Tensor):
    batch, token_count, _ = x.shape
    qkv = mixer.qkv(x).reshape(
        batch,
        token_count,
        3,
        mixer.num_heads,
        mixer.head_dim,
    )
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    gradient = -(mixer.scale / float(token_count)) * (
        k.transpose(-2, -1) @ v
    )
    normalized_gradient = gradient / (
        gradient.norm(dim=-2, keepdim=True) + 1.0
    )
    fast_weights = mixer.w0 - mixer.inner_lr * normalized_gradient
    output = (q @ fast_weights).transpose(1, 2).reshape(
        batch, token_count, mixer.dim
    )
    return output, fast_weights, gradient, normalized_gradient


def assert_reference_formula_and_fast_weight_lifecycle():
    torch.manual_seed(3)
    mixer = GlobalLinearTTTMixer(
        dim=8,
        num_heads=2,
        qkv_bias=True,
        inner_lr=0.75,
    ).eval()
    x = torch.randn(2, 6, 8)
    w0_before = mixer.w0.detach().clone()

    with torch.no_grad():
        expected, expected_fast_weights, gradient, normalized_gradient = (
            explicit_reference(mixer, x)
        )
        actual = mixer(x, height=2, width=3)
        repeated = mixer(x, height=2, width=3)

    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(repeated, actual, atol=0.0, rtol=0.0)
    torch.testing.assert_close(mixer.w0, w0_before, atol=0.0, rtol=0.0)
    assert expected_fast_weights.shape == (2, 2, 4, 4)
    assert not torch.allclose(expected_fast_weights[0], expected_fast_weights[1])

    whole_matrix_normalization = gradient / (
        gradient.norm(dim=(-2, -1), keepdim=True) + 1.0
    )
    assert not torch.allclose(normalized_gradient, whole_matrix_normalization)


def assert_backward_and_initialization():
    torch.manual_seed(5)
    cpe = DepthwiseCPE2D(dim=8)
    mixer = GlobalLinearTTTMixer(dim=8, num_heads=2)
    x = torch.randn(2, 8, 2, 3, requires_grad=True)
    spatial = x + cpe(x, periodic=True)
    tokens = spatial.permute(0, 2, 3, 1).reshape(2, 6, 8)
    output = mixer(tokens, height=2, width=3, periodic=True)
    output.square().mean().backward()

    parameters = (cpe.proj.weight, mixer.qkv.weight, mixer.w0)
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert all(parameter.grad.abs().sum().item() > 0.0 for parameter in parameters)
    assert 0.01 < mixer.w0.detach().float().std().item() < 0.03
    assert 0.01 < mixer.qkv.weight.detach().float().std().item() < 0.03
    assert mixer.qkv.bias is not None
    torch.testing.assert_close(
        mixer.qkv.bias,
        torch.zeros_like(mixer.qkv.bias),
        atol=0.0,
        rtol=0.0,
    )


def assert_periodic_cpe_boundary():
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


def assert_full_model_integration():
    torch.manual_seed(7)
    model = PDETransformer(
        sample_size=128,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        periodic=True,
        carrier_token_active=False,
        token_mixer_type="global_linear_ttt",
        vittt_inner_lr=1.0,
        vittt_head_dim=32,
    ).eval()

    mixers = [
        module for module in model.modules()
        if isinstance(module, GlobalLinearTTTMixer)
    ]
    cpes = [module for module in model.modules() if isinstance(module, DepthwiseCPE2D)]
    conv_mlps = [
        module for module in model.modules() if isinstance(module, ConvEnhancedMlp)
    ]
    assert len(mixers) == 22
    assert len(cpes) == 22
    assert not conv_mlps
    assert len({id(module.w0) for module in mixers}) == 22
    assert len({id(module.proj.weight) for module in cpes}) == 22
    assert all(module.head_dim == 32 for module in mixers)
    expected_heads = {96: 3, 192: 6, 384: 12}
    assert all(module.num_heads == expected_heads[module.dim] for module in mixers)
    assert all(
        0.015 < module.w0.detach().float().std().item() < 0.025
        for module in mixers
    )
    assert all(
        0.015 < module.qkv.weight.detach().float().std().item() < 0.025
        for module in mixers
    )

    calls = []
    original_forward = GlobalLinearTTTMixer.forward

    def recording_forward(self, x, height, width, periodic=False):
        calls.append((x.shape[1], x.shape[2], height, width))
        return original_forward(self, x, height, width, periodic)

    def forbidden(*args, **kwargs):
        raise AssertionError("Global Linear TTT must not use window or causal operations.")

    x = torch.randn(1, 2, 128, 128)
    t = torch.zeros(1)
    y = torch.zeros(1, dtype=torch.long)
    with (
        patch.object(GlobalLinearTTTMixer, "forward", recording_forward),
        patch.object(pde_module, "window_partition", forbidden),
        patch.object(pde_module, "window_reverse", forbidden),
        patch.object(pde_module.torch, "roll", forbidden),
        patch.object(pde_module.torch, "tril", forbidden),
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


def assert_no_window_or_causal_source():
    source = inspect.getsource(linear_ttt_module)
    forbidden = ("window_partition", "window_reverse", "torch.roll", "torch.tril")
    assert all(name not in source for name in forbidden)


def assert_pretrained_round_trip():
    model_type = "PDE-TINY-GLOBAL-LINEAR-TTT-SAVE-LOAD"

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
            token_mixer_type="global_linear_ttt",
            vittt_inner_lr=0.375,
            vittt_head_dim=16,
        ).eval()
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            loaded = PDETransformer.from_pretrained(directory).eval()

        expected_config = {
            "token_mixer_type": "global_linear_ttt",
            "vittt_inner_lr": 0.375,
            "vittt_head_dim": 16,
        }
        for name, value in expected_config.items():
            assert getattr(loaded.config, name) == value

        loaded_mixers = [
            module for module in loaded.modules()
            if isinstance(module, GlobalLinearTTTMixer)
        ]
        assert loaded_mixers
        assert all(module.head_dim == 16 for module in loaded_mixers)
        assert all(module.inner_lr == 0.375 for module in loaded_mixers)

        expected_state = model.state_dict()
        loaded_state = loaded.state_dict()
        assert expected_state.keys() == loaded_state.keys()
        for name in expected_state:
            torch.testing.assert_close(
                loaded_state[name], expected_state[name], atol=0.0, rtol=0.0
            )

        probe = torch.randn(2, 4, 32)
        with torch.no_grad():
            expected = next(
                module for module in model.modules()
                if isinstance(module, GlobalLinearTTTMixer)
            )(probe, height=2, width=2)
            actual = loaded_mixers[0](probe, height=2, width=2)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    finally:
        del pde_module.PDE_models[model_type]


def assert_matched_100_epoch_config():
    plain = OmegaConf.to_container(OmegaConf.load(PLAIN_CONFIG), resolve=True)
    linear = OmegaConf.to_container(OmegaConf.load(LINEAR_CONFIG), resolve=True)
    assert plain["token_mixer_type"] == "attention"
    assert linear["token_mixer_type"] == "global_linear_ttt"
    for config in (plain, linear):
        config.pop("run_name")
        config.pop("token_mixer_type")
    assert plain == linear


def assert_training_entrypoint_accepts_config():
    mixed_channels = sys.modules["pdetransformer.core.mixed_channels"]
    mixed_channels.PDETransformer = pde_module.PDETransformer
    mixed_channels.SingleStepSupervised = importlib.import_module(
        "pdetransformer.core.mixed_channels.train_supervised"
    ).SingleStepSupervised
    entrypoint_path = (
        REPO_ROOT / "server_example" / "train_global_vittt_ape_xxl_server.py"
    )
    spec = importlib.util.spec_from_file_location("global_ttt_train_entrypoint", entrypoint_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = module.load_config(LINEAR_CONFIG)
    assert config["token_mixer_type"] == "global_linear_ttt"


def assert_true_half_precision_when_supported():
    if not torch.cuda.is_available():
        return
    dtypes = [torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    for dtype in dtypes:
        mixer = GlobalLinearTTTMixer(dim=16, num_heads=4).cuda().to(dtype=dtype)
        x = torch.randn(2, 16, 16, device="cuda", dtype=dtype, requires_grad=True)
        output = mixer(x, height=4, width=4, periodic=True)
        assert output.dtype == dtype
        assert torch.isfinite(output).all()
        output.float().square().mean().backward()
        assert mixer.w0.grad is not None
        assert torch.isfinite(mixer.w0.grad).all()


def main():
    checks = [
        (
            "normalized reference formula and fast-weight lifecycle",
            assert_reference_formula_and_fast_weight_lifecycle,
        ),
        ("CPU FP32 backward and initialization", assert_backward_and_initialization),
        ("periodic CPE boundary", assert_periodic_cpe_boundary),
        ("no window or causal source", assert_no_window_or_causal_source),
        ("full-model token integration", assert_full_model_integration),
        ("Diffusers save/load round trip", assert_pretrained_round_trip),
        ("matched 100-epoch configuration", assert_matched_100_epoch_config),
        ("training entrypoint configuration", assert_training_entrypoint_accepts_config),
        ("CUDA true half precision", assert_true_half_precision_when_supported),
    ]
    for name, check in checks:
        check()
        print(f"PASS: {name}")


if __name__ == "__main__":
    main()
