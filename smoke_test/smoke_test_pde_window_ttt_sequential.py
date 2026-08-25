"""Checks the controlled sequential schedules for window Linear/MLP TTT."""

import importlib
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "server_example"
CONFIGS = {
    "linear_token": CONFIG_DIR / "pdes_window-linear-token-sequential_128_60sims.yaml",
    "linear_window": CONFIG_DIR / "pdes_window-linear-window-sequential_128_60sims.yaml",
    "mlp_token": CONFIG_DIR / "pdes_window-mlp-token-sequential_128_60sims.yaml",
    "mlp_window": CONFIG_DIR / "pdes_window-mlp-window-sequential_128_60sims.yaml",
}


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
    mlp = importlib.import_module("pdetransformer.core.pde_vittt_window_mlp")
    return pde, linear, mlp


pde_module, linear_module, mlp_module = load_experiment_modules()
PDETransformer = pde_module.PDETransformer
GlobalLinearTTTMixer = linear_module.GlobalLinearTTTMixer
WindowFullBatchMLPTTTMixer = mlp_module.WindowFullBatchMLPTTTMixer


def project_qkv(mixer, x):
    batch, token_count, _ = x.shape
    qkv = mixer.qkv(x).reshape(
        batch, token_count, 3, mixer.num_heads, mixer.head_dim
    )
    return qkv.permute(2, 0, 3, 1, 4).unbind(0)


def linear_update(mixer, k, v, weights):
    gradient = -(mixer.scale / float(k.shape[-2])) * (
        k.transpose(-2, -1) @ v
    )
    gradient = gradient / (gradient.norm(dim=-2, keepdim=True) + 1.0)
    return weights - mixer.inner_lr * gradient


def mlp_update(mixer, k, v, weights):
    w1, w2 = weights
    z1 = k @ w1
    hidden = F.gelu(z1)
    output_gradient = -(mixer.scale / float(k.shape[-2])) * v
    gradient_w2 = hidden.transpose(-2, -1) @ output_gradient
    gradient_hidden = output_gradient @ w2.transpose(-2, -1)
    gradient_w1 = k.transpose(-2, -1) @ (
        gradient_hidden * mixer._gelu_derivative(z1)
    )
    gradient_w1 = gradient_w1 / (
        gradient_w1.norm(dim=-2, keepdim=True) + 1.0
    )
    gradient_w2 = gradient_w2 / (
        gradient_w2.norm(dim=-2, keepdim=True) + 1.0
    )
    return (
        w1 - mixer.inner_lr * gradient_w1,
        w2 - mixer.inner_lr * gradient_w2,
    )


def apply_mlp(q, weights):
    w1, w2 = weights
    return F.gelu(q @ w1) @ w2


def explicit_token_sequential(mixer, x):
    q, k, v = project_qkv(mixer, x)
    batch = x.shape[0]
    if isinstance(mixer, GlobalLinearTTTMixer):
        weights = mixer.w0.expand(batch, -1, -1, -1)
        update = linear_update
        apply = lambda query, current: query @ current
    else:
        weights = (
            mixer.w1.expand(batch, -1, -1, -1),
            mixer.w2.expand(batch, -1, -1, -1),
        )
        update = mlp_update
        apply = apply_mlp
    outputs = []
    for start in range(0, x.shape[1], mixer.chunk_size):
        stop = min(start + mixer.chunk_size, x.shape[1])
        weights = update(mixer, k[..., start:stop, :], v[..., start:stop, :], weights)
        outputs.append(apply(q[..., start:stop, :], weights))
    output = torch.cat(outputs, dim=-2)
    return output.transpose(1, 2).reshape_as(x)


def explicit_window_sequential(mixer, x, windows_per_sample):
    q, k, v = project_qkv(mixer, x)
    window_batch, num_heads, token_count, head_dim = q.shape
    sample_batch = window_batch // windows_per_sample
    q = q.reshape(sample_batch, windows_per_sample, num_heads, token_count, head_dim)
    k = k.reshape_as(q)
    v = v.reshape_as(q)
    if isinstance(mixer, GlobalLinearTTTMixer):
        weights = mixer.w0.expand(sample_batch, -1, -1, -1)
        update = linear_update
        apply = lambda query, current: query @ current
    else:
        weights = (
            mixer.w1.expand(sample_batch, -1, -1, -1),
            mixer.w2.expand(sample_batch, -1, -1, -1),
        )
        update = mlp_update
        apply = apply_mlp
    outputs = []
    for window_index in range(windows_per_sample):
        weights = update(
            mixer, k[:, window_index], v[:, window_index], weights
        )
        outputs.append(apply(q[:, window_index], weights))
    output = torch.stack(outputs, dim=1).reshape(
        window_batch, num_heads, token_count, head_dim
    )
    return output.transpose(1, 2).reshape_as(x)


def assert_reference_schedules():
    torch.manual_seed(12)
    factories = (
        lambda mode: GlobalLinearTTTMixer(
            dim=8, num_heads=2, inner_lr=0.6, update_mode=mode, chunk_size=2
        ),
        lambda mode: WindowFullBatchMLPTTTMixer(
            dim=8,
            num_heads=2,
            inner_lr=0.6,
            hidden_ratio=2,
            update_mode=mode,
            chunk_size=2,
        ),
    )
    x = torch.randn(6, 4, 8)
    for factory in factories:
        token_mixer = factory("token_sequential").eval()
        expected = explicit_token_sequential(token_mixer, x)
        actual = token_mixer(x, height=2, width=2, windows_per_sample=3)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)

        window_mixer = factory("window_sequential").eval()
        expected = explicit_window_sequential(window_mixer, x, windows_per_sample=3)
        actual = window_mixer(x, height=2, width=2, windows_per_sample=3)
        repeated = window_mixer(x, height=2, width=2, windows_per_sample=3)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(actual, repeated, atol=0.0, rtol=0.0)


def assert_window_state_scope():
    torch.manual_seed(21)
    x = torch.randn(6, 4, 8)
    for mixer in (
        GlobalLinearTTTMixer(
            dim=8, num_heads=2, update_mode="window_sequential"
        ),
        WindowFullBatchMLPTTTMixer(
            dim=8, num_heads=2, update_mode="window_sequential", hidden_ratio=2
        ),
    ):
        baseline = mixer(x, height=2, width=2, windows_per_sample=3)

        changed_earlier_window = x.clone()
        changed_earlier_window[0] += 2.0
        changed = mixer(
            changed_earlier_window, height=2, width=2, windows_per_sample=3
        )
        assert not torch.allclose(baseline[1], changed[1])
        torch.testing.assert_close(baseline[3:], changed[3:], atol=0.0, rtol=0.0)


def assert_backward_all_modes():
    for mixer_class in (GlobalLinearTTTMixer, WindowFullBatchMLPTTTMixer):
        for mode in ("token_sequential", "window_sequential"):
            mixer = mixer_class(dim=8, num_heads=2, update_mode=mode, chunk_size=2)
            x = torch.randn(4, 4, 8, requires_grad=True)
            output = mixer(x, height=2, width=2, windows_per_sample=2)
            output.square().mean().backward()
            assert mixer.qkv.weight.grad is not None
            initial_parameters = (
                (mixer.w0,)
                if isinstance(mixer, GlobalLinearTTTMixer)
                else (mixer.w1, mixer.w2)
            )
            assert all(parameter.grad is not None for parameter in initial_parameters)


def assert_model_integration_and_save_load():
    model_type = "PDE-TINY-WINDOW-SEQUENTIAL-SAVE-LOAD"

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
            sample_size=64,
            in_channels=2,
            out_channels=2,
            type=model_type,
            periodic=True,
            carrier_token_active=False,
            token_mixer_type="window_linear_ttt",
            vittt_inner_lr=0.375,
            vittt_head_dim=16,
            window_ttt_update_mode="window_sequential",
            window_ttt_chunk_size=8,
        ).eval()
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            output = model(
                x,
                timestep=torch.zeros(1),
                class_labels=torch.zeros(1, dtype=torch.long),
            ).sample
        assert output.shape == x.shape

        with TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            loaded = PDETransformer.from_pretrained(directory).eval()
        assert loaded.config.window_ttt_update_mode == "window_sequential"
        assert loaded.config.window_ttt_chunk_size == 8
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                loaded.state_dict()[name], value, atol=0.0, rtol=0.0
            )
    finally:
        del pde_module.PDE_models[model_type]


def assert_configs():
    loaded = {
        name: OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        for name, path in CONFIGS.items()
    }
    assert loaded["linear_token"]["window_ttt_update_mode"] == "token_sequential"
    assert loaded["linear_window"]["window_ttt_update_mode"] == "window_sequential"
    assert loaded["mlp_token"]["window_ttt_update_mode"] == "token_sequential"
    assert loaded["mlp_window"]["window_ttt_update_mode"] == "window_sequential"
    assert all(config["window_ttt_chunk_size"] == 16 for config in loaded.values())


def main():
    checks = [
        ("Linear/MLP explicit sequential references", assert_reference_schedules),
        ("cross-window state scope", assert_window_state_scope),
        ("CPU FP32 backward for four schedules", assert_backward_all_modes),
        ("full-model integration and save/load", assert_model_integration_and_save_load),
        ("four matched YAML configs", assert_configs),
    ]
    for name, check in checks:
        check()
        print(f"PASS: {name}")


if __name__ == "__main__":
    main()
