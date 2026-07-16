from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import lightning
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
for package_name, package_path in {
    "pdetransformer": REPO_ROOT / "pdetransformer",
    "pdetransformer.core": REPO_ROOT / "pdetransformer" / "core",
}.items():
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

from pdetransformer.core.mixed_channels import pde_transformer as pde_module
from pdetransformer.core.mixed_channels.pde_transformer import PDETransformer
from pdetransformer.core.mixed_channels.train_supervised import (
    SingleStepSupervised,
    TemporalRolloutSupervised,
    _detach_state_tree,
)
from pdetransformer.core.pde_temporal_ttt import PDETemporalTTT2D
from server_example.train_global_linear_temporal_server import (
    freeze_gl_backbone,
    initialize_from_gl_checkpoint,
    load_config,
    parameter_summary,
)


TRAIN_CONFIG = (
    REPO_ROOT
    / "server_example"
    / "pdes_global-linear-temporal-ttt-frozenbase_128_100ep_60sims.yaml"
)


def _tiny_model_type() -> str:
    model_type = "PDE-TEMPORAL-SMOKE"

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
    return model_type


def _model_args(model_type: str) -> dict:
    return {
        "sample_size": 32,
        "in_channels": 2,
        "out_channels": 2,
        "type": model_type,
        "patch_size": 4,
        "periodic": True,
        "carrier_token_active": False,
        "token_mixer_type": "global_linear_ttt",
        "vittt_inner_lr": 1.0,
        "vittt_head_dim": 32,
    }


def test_temporal_module_state_and_gradients() -> None:
    torch.manual_seed(0)
    module = PDETemporalTTT2D(
        dim=32,
        num_heads=4,
        layer_type="mlp",
        mini_batch_size=16,
        gate_init=0.1,
    )
    x = torch.randn(2, 32, 4, 4, requires_grad=True)
    y1, state1 = module(x)
    y2, state2 = module(x, state=state1)

    assert y1.shape == x.shape
    assert y2.shape == x.shape
    assert not torch.equal(y1, y2)
    assert not torch.equal(state1["W1_states"], state2["W1_states"])
    y2.square().mean().backward()
    assert module.gate.grad is not None
    assert module.mixer.ttt.W1.grad is not None
    detached = _detach_state_tree(state2)
    assert all(not value.requires_grad for value in detached.values())


def test_gl_checkpoint_bypass_freeze_and_save_load() -> None:
    model_type = _tiny_model_type()
    torch.manual_seed(1)
    base_model = PDETransformer(**_model_args(model_type))
    base_strategy = SingleStepSupervised(model=base_model)

    torch.manual_seed(2)
    temporal_model = PDETransformer(
        **_model_args(model_type),
        temporal_ttt_enabled=True,
        temporal_ttt_layer_type="mlp",
        temporal_ttt_mini_batch_size=16,
        temporal_ttt_gate_init=0.1,
    )
    temporal_strategy = TemporalRolloutSupervised(
        model=temporal_model,
        train_unrolling_steps=4,
        tbptt_chunk_size=2,
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "gl.ckpt"
        torch.save({"state_dict": base_strategy.state_dict()}, checkpoint)
        initialize_from_gl_checkpoint(temporal_strategy, checkpoint)
        freeze_gl_backbone(temporal_strategy)
        total, trainable = parameter_summary(temporal_strategy)
        assert 0 < trainable < total
        assert all(
            (not parameter.requires_grad) or ".temporal_ttt." in name
            for name, parameter in temporal_strategy.named_parameters()
        )

        x = torch.randn(1, 2, 32, 32)
        labels = torch.zeros(1, dtype=torch.long)
        base_model.eval()
        temporal_model.eval()
        with torch.no_grad():
            base_output = base_model(x, class_labels=labels).sample
            bypass_output = temporal_model(
                x,
                class_labels=labels,
                use_temporal_ttt=False,
            ).sample
            active1 = temporal_model(
                x,
                class_labels=labels,
                ttt_state_cache={},
                return_ttt_state_cache=True,
            )
            active2 = temporal_model(
                x,
                class_labels=labels,
                ttt_state_cache=active1.ttt_state_cache,
                return_ttt_state_cache=True,
            )

        assert torch.equal(base_output, bypass_output)
        assert active1.sample.shape == base_output.shape
        assert "temporal_latent" in active1.ttt_state_cache
        state1 = active1.ttt_state_cache["temporal_latent"]["W1_states"]
        state2 = active2.ttt_state_cache["temporal_latent"]["W1_states"]
        assert not torch.equal(state1, state2)

        save_dir = root / "pretrained"
        temporal_model.save_pretrained(save_dir)
        restored = PDETransformer.from_pretrained(save_dir)
        restored.eval()
        assert restored.config.token_mixer_type == "global_linear_ttt"
        assert restored.config.temporal_ttt_enabled is True
        with torch.no_grad():
            restored_bypass = restored(
                x,
                class_labels=labels,
                use_temporal_ttt=False,
            ).sample
        assert torch.equal(bypass_output, restored_bypass)

    del pde_module.PDE_models[model_type]


def test_real_pde_s_latent_dimensions() -> None:
    torch.manual_seed(3)
    model = PDETransformer(
        sample_size=128,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        patch_size=4,
        periodic=True,
        carrier_token_active=False,
        token_mixer_type="global_linear_ttt",
        vittt_head_dim=32,
        temporal_ttt_enabled=True,
        temporal_ttt_layer_type="mlp",
        temporal_ttt_mini_batch_size=64,
    )
    seen = []

    def capture(module, args):
        seen.append(tuple(args[0].shape))

    handle = model.model.temporal_ttt.register_forward_pre_hook(capture)
    with torch.no_grad():
        output = model(
            torch.randn(1, 2, 128, 128),
            class_labels=torch.zeros(1, dtype=torch.long),
            ttt_state_cache={},
            return_ttt_state_cache=True,
        )
    handle.remove()
    assert seen == [(1, 384, 8, 8)]
    assert output.sample.shape == (1, 2, 128, 128)


def test_bypass_rejects_missing_temporal_module() -> None:
    model_type = _tiny_model_type()
    model = PDETransformer(**_model_args(model_type))
    try:
        model(
            torch.randn(1, 2, 32, 32),
            class_labels=torch.zeros(1, dtype=torch.long),
            use_temporal_ttt=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Explicit temporal use must reject a model without the module")
    del pde_module.PDE_models[model_type]


def test_training_config() -> None:
    config = load_config(TRAIN_CONFIG)
    assert config["token_mixer_type"] == "global_linear_ttt"
    assert config["temporal_ttt_enabled"] is True
    assert config["freeze_backbone"] is True
    assert config["train_unrolling_steps"] == 29
    assert config["train_step_size"] == 29
    assert config["tbptt_chunk_size"] == 4
    assert config["accumulate_grad_batches"] == 8


class _ToySequenceDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict:
        data = torch.linspace(0.1, 0.5, 5).view(5, 1, 1, 1).expand(5, 1, 4, 4)
        return {
            "data": data,
            "loading_metadata": {},
            "physical_metadata": {"PDE": torch.tensor([0])},
        }


class _ToyStatefulModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal_ttt = torch.nn.ParameterDict(
            {"scale": torch.nn.Parameter(torch.tensor(0.8))}
        )

    def forward(
        self,
        x: torch.Tensor,
        class_labels=None,
        ttt_state_cache=None,
        return_ttt_state_cache: bool = False,
    ) -> SimpleNamespace:
        state = (ttt_state_cache or {}).get("temporal_latent")
        if state is None:
            state = torch.zeros_like(x)
        scale = self.temporal_ttt["scale"]
        next_state = state + scale * x
        sample = scale * x + 0.01 * next_state
        return SimpleNamespace(
            sample=sample,
            ttt_state_cache={"temporal_latent": next_state},
        )


def test_lightning_tbptt_step() -> None:
    model = _ToyStatefulModel()
    initial_scale = model.temporal_ttt["scale"].detach().clone()
    strategy = TemporalRolloutSupervised(
        model=model,
        train_unrolling_steps=4,
        tbptt_chunk_size=2,
        gradient_accumulation_batches=2,
    )
    strategy.learning_rate = 1e-2
    loader = DataLoader(_ToySequenceDataset(), batch_size=1)
    trainer = lightning.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(strategy, train_dataloaders=loader, val_dataloaders=loader)
    assert not torch.equal(initial_scale, model.temporal_ttt["scale"].detach())


def test_partial_accumulation_rescaling() -> None:
    model = _ToyStatefulModel()
    strategy = TemporalRolloutSupervised(
        model=model,
        train_unrolling_steps=4,
        tbptt_chunk_size=2,
        gradient_accumulation_batches=4,
    )
    parameter = model.temporal_ttt["scale"]
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    initial = parameter.detach().clone()
    parameter.grad = torch.full_like(parameter, 0.25)
    strategy._batches_since_optimizer_step = 1
    strategy._optimizer_step_if_ready(optimizer, force=True)
    torch.testing.assert_close(parameter.detach(), initial - 0.1)
    assert strategy._batches_since_optimizer_step == 0
    assert strategy._batches_since_optimizer_step == 0


if __name__ == "__main__":
    test_temporal_module_state_and_gradients()
    print("PASS: temporal state and gradients")
    test_gl_checkpoint_bypass_freeze_and_save_load()
    print("PASS: G-L checkpoint, bypass, freeze, and save/load")
    test_real_pde_s_latent_dimensions()
    print("PASS: real PDE-S latent dimensions")
    test_bypass_rejects_missing_temporal_module()
    print("PASS: explicit temporal guard")
    test_training_config()
    print("PASS: matched G-L-TF100 configuration")
    test_lightning_tbptt_step()
    print("PASS: Lightning TBPTT step")
    test_partial_accumulation_rescaling()
    print("PASS: partial accumulation rescaling")
