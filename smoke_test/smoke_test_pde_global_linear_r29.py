from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from lightning import Trainer
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name, package_path in {
    "pdetransformer": ROOT / "pdetransformer",
    "pdetransformer.core": ROOT / "pdetransformer" / "core",
}.items():
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

from pdetransformer.core.mixed_channels import (  # noqa: E402
    AutoregressiveRolloutSupervised,
)
from pdetransformer.core.pde_vittt_global_linear import (  # noqa: E402
    GlobalLinearTTTMixer,
)
from server_example.train_global_linear_rollout_server import (  # noqa: E402
    build_training_module,
    load_config,
)


CONFIG = ROOT / "server_example" / "pdes_global-linear-r29_128_100ep_60sims.yaml"


class ScalarRolloutModel(nn.Module):
    def __init__(self, value: float = 0.5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value))

    def forward(self, x: torch.Tensor, class_labels=None):
        del class_labels
        return SimpleNamespace(sample=x * self.scale)


def test_config_and_model() -> None:
    config = load_config(CONFIG)
    assert config["train_unrolling_steps"] == 29
    assert config["tbptt_chunk_size"] == 4
    assert config["token_mixer_type"] == "global_linear_ttt"
    module = build_training_module(config)
    mixers = [
        child for child in module.modules() if isinstance(child, GlobalLinearTTTMixer)
    ]
    assert len(mixers) == 22
    assert not any(hasattr(mixer, "persistent_state") for mixer in mixers)
    assert sum(parameter.numel() for parameter in module.parameters()) == 33_361_952


def test_chunk_keeps_prediction_graph() -> None:
    model = ScalarRolloutModel(0.5)
    module = AutoregressiveRolloutSupervised(
        model=model,
        train_unrolling_steps=4,
        tbptt_chunk_size=4,
    )
    input_frame = torch.ones(1, 1, 1, 1)
    targets = torch.zeros(1, 4, 1, 1, 1)
    labels = torch.zeros(1, dtype=torch.long)
    prediction, loss = module._rollout_chunk(
        input_frame,
        targets,
        labels,
        0,
        4,
    )
    loss.backward()

    scale = 0.5
    expected_loss = sum(scale ** (2 * step) for step in range(1, 5))
    expected_grad = sum(
        2 * step * scale ** (2 * step - 1) for step in range(1, 5)
    )
    torch.testing.assert_close(loss, torch.tensor(expected_loss))
    torch.testing.assert_close(model.scale.grad, torch.tensor(expected_grad))
    torch.testing.assert_close(prediction, torch.tensor([[[[scale**4]]]]))


def test_chunk_boundary_can_detach() -> None:
    model = ScalarRolloutModel(0.5)
    module = AutoregressiveRolloutSupervised(
        model=model,
        train_unrolling_steps=4,
        tbptt_chunk_size=2,
    )
    input_frame = torch.ones(1, 1, 1, 1)
    targets = torch.zeros(1, 4, 1, 1, 1)
    labels = torch.zeros(1, dtype=torch.long)
    prediction, first_loss = module._rollout_chunk(
        input_frame, targets, labels, 0, 2
    )
    first_loss.backward()
    first_grad = model.scale.grad.detach().clone()

    model.scale.grad = None
    prediction, second_loss = module._rollout_chunk(
        prediction.detach(), targets, labels, 2, 4
    )
    second_loss.backward()
    assert first_grad.item() > 0
    assert model.scale.grad is not None
    assert prediction.grad_fn is not None


def test_lightning_manual_optimization() -> None:
    model = ScalarRolloutModel(0.5)
    initial_scale = model.scale.detach().clone()
    module = AutoregressiveRolloutSupervised(
        model=model,
        train_unrolling_steps=4,
        tbptt_chunk_size=2,
        gradient_accumulation_batches=1,
    )
    module.learning_rate = 0.01
    sample = {
        "data": torch.tensor([1.0, 0.4, 0.2, 0.1, 0.05]).reshape(5, 1, 1, 1),
        "loading_metadata": {},
        "physical_metadata": {"PDE": torch.tensor([0])},
    }
    loader = DataLoader([sample], batch_size=1)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert not torch.equal(initial_scale, model.scale.detach())


if __name__ == "__main__":
    test_config_and_model()
    print("PASS: config and 22 stateless spatial mixers")
    test_chunk_keeps_prediction_graph()
    print("PASS: four-step prediction graph")
    test_chunk_boundary_can_detach()
    print("PASS: TBPTT chunk boundary")
    test_lightning_manual_optimization()
    print("PASS: Lightning manual optimization")
