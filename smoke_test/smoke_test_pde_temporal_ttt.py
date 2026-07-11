from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import lightning
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pdetransformer.core.mixed_channels.pde_transformer import PDETransformer
from pdetransformer.core.mixed_channels.train_supervised import (
    TemporalRolloutSupervised,
    _detach_state_tree,
)
from pdetransformer.core.pde_temporal_ttt import PDETemporalTTT2D


def test_temporal_module() -> None:
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
    assert set(state1) == {
        "W1_states",
        "b1_states",
        "W2_states",
        "b2_states",
        "W1_grad",
        "b1_grad",
        "W2_grad",
        "b2_grad",
    }
    assert not torch.equal(y1, y2)
    assert not torch.equal(state1["W1_states"], state2["W1_states"])

    y2.square().mean().backward()
    assert module.gate.grad is not None
    assert module.mixer.ttt.W1.grad is not None

    detached = _detach_state_tree(state2)
    assert all(not value.requires_grad for value in detached.values())


def test_full_model_checkpoint_compatibility() -> None:
    torch.manual_seed(1)
    model_args = dict(
        sample_size=64,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        patch_size=4,
        periodic=True,
        carrier_token_active=False,
        token_mixer_type="attention",
    )
    base = PDETransformer(**model_args)
    model = PDETransformer(
        **model_args,
        temporal_ttt_enabled=True,
        temporal_ttt_layer_type="mlp",
        temporal_ttt_mini_batch_size=16,
        temporal_ttt_gate_init=0.0,
    )
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(".temporal_ttt." in key for key in missing)
    assert not unexpected

    x = torch.randn(1, 2, 64, 64)
    labels = torch.zeros(1, dtype=torch.long)
    base.eval()
    model.eval()
    with torch.no_grad():
        base_output = base(x, class_labels=labels).sample
        output1 = model(
            x,
            class_labels=labels,
            ttt_state_cache={},
            return_ttt_state_cache=True,
        )
        output2 = model(
            x,
            class_labels=labels,
            ttt_state_cache=output1.ttt_state_cache,
            return_ttt_state_cache=True,
        )

    assert torch.equal(base_output, output1.sample)
    assert torch.equal(base_output, output2.sample)
    assert "temporal_latent" in output2.ttt_state_cache
    state1 = output1.ttt_state_cache["temporal_latent"]["W1_states"]
    state2 = output2.ttt_state_cache["temporal_latent"]["W1_states"]
    assert not torch.equal(state1, state2)


def test_sequence_validation() -> None:
    model = torch.nn.Identity()
    strategy = TemporalRolloutSupervised(
        model=model,
        train_unrolling_steps=4,
        tbptt_chunk_size=2,
    )
    target = torch.zeros(2, 4, 1, 8, 8)
    assert strategy._validate_sequence_length(target) == 4
    try:
        strategy._validate_sequence_length(target[:, :3])
    except ValueError:
        pass
    else:
        raise AssertionError("short sequences must be rejected")


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
    assert strategy._batches_since_optimizer_step == 0


if __name__ == "__main__":
    test_temporal_module()
    test_full_model_checkpoint_compatibility()
    test_sequence_validation()
    test_lightning_tbptt_step()
    print("PDE temporal TTT smoke test passed")
