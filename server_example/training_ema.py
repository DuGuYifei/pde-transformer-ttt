"""Checkpoint-safe exponential moving averages for focused server training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch


EMA_STATE_VERSION = 1


class TrainingEMA(L.Callback):
    """Track trainable parameters and optionally validate with their EMA copy."""

    def __init__(
        self,
        decay: float = 0.999,
        update_every_n_steps: int = 1,
        validate_with_ema: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")
        if update_every_n_steps < 1:
            raise ValueError("EMA update_every_n_steps must be positive.")
        self.decay = float(decay)
        self.update_every_n_steps = int(update_every_n_steps)
        self.validate_with_ema = bool(validate_with_ema)
        self.num_updates = 0
        self._last_global_step = 0
        self._shadow: dict[str, torch.Tensor] = {}
        self._raw_backup: dict[str, torch.Tensor] = {}
        self._using_ema = False

    @property
    def state_key(self) -> str:
        return "TrainingEMA"

    @staticmethod
    def _trainable_parameters(pl_module: L.LightningModule):
        return (
            (name, parameter)
            for name, parameter in pl_module.named_parameters()
            if parameter.requires_grad
        )

    @torch.no_grad()
    def _initialize(self, pl_module: L.LightningModule) -> None:
        self._shadow = {
            name: parameter.detach().clone()
            for name, parameter in self._trainable_parameters(pl_module)
        }

    def on_fit_start(self, trainer, pl_module) -> None:
        parameters = dict(self._trainable_parameters(pl_module))
        if not self._shadow:
            self._initialize(pl_module)
        else:
            missing = sorted(set(parameters) - set(self._shadow))
            unexpected = sorted(set(self._shadow) - set(parameters))
            if missing or unexpected:
                raise RuntimeError(
                    "EMA checkpoint parameters do not match the model: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            self._shadow = {
                name: value.to(device=parameters[name].device, dtype=parameters[name].dtype)
                for name, value in self._shadow.items()
            }
        self._last_global_step = int(trainer.global_step)

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        global_step = int(trainer.global_step)
        if global_step <= self._last_global_step:
            return
        self._last_global_step = global_step
        if global_step % self.update_every_n_steps != 0:
            return

        parameters = list(self._trainable_parameters(pl_module))
        shadow_values = [self._shadow[name] for name, _ in parameters]
        current_values = [parameter.detach() for _, parameter in parameters]
        torch._foreach_mul_(shadow_values, self.decay)
        torch._foreach_add_(
            shadow_values,
            current_values,
            alpha=1.0 - self.decay,
        )
        self.num_updates += 1

    @torch.no_grad()
    def _swap_to_ema(self, pl_module: L.LightningModule) -> None:
        if self._using_ema:
            raise RuntimeError("EMA weights are already active.")
        self._raw_backup = {}
        for name, parameter in self._trainable_parameters(pl_module):
            self._raw_backup[name] = parameter.detach().clone()
            parameter.copy_(self._shadow[name])
        self._using_ema = True

    @torch.no_grad()
    def _restore_raw(self, pl_module: L.LightningModule) -> None:
        if not self._using_ema:
            return
        for name, parameter in self._trainable_parameters(pl_module):
            parameter.copy_(self._raw_backup[name])
        self._raw_backup = {}
        self._using_ema = False

    def functional_parameters_for(
        self,
        module: torch.nn.Module,
        *,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        """Return EMA parameters keyed for ``torch.func.functional_call``."""

        if not self._shadow:
            raise RuntimeError("EMA parameters are not initialized.")
        expected = set(dict(module.named_parameters()))
        selected = {
            name.removeprefix(prefix): value
            for name, value in self._shadow.items()
            if name.startswith(prefix)
        }
        missing = sorted(expected - set(selected))
        unexpected = sorted(set(selected) - expected)
        if missing or unexpected:
            raise RuntimeError(
                "EMA functional parameters do not match the target module: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return selected

    def on_validation_start(self, trainer, pl_module) -> None:
        if self.validate_with_ema and self._shadow:
            self._swap_to_ema(pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        self._restore_raw(pl_module)

    def on_exception(self, trainer, pl_module, exception) -> None:
        self._restore_raw(pl_module)

    def state_dict(self) -> dict[str, Any]:
        if self._using_ema:
            raise RuntimeError("Cannot checkpoint while EMA weights are swapped into the model.")
        return {
            "ema_state_version": EMA_STATE_VERSION,
            "decay": self.decay,
            "update_every_n_steps": self.update_every_n_steps,
            "validate_with_ema": self.validate_with_ema,
            "num_updates": self.num_updates,
            "shadow": {
                name: value.detach().cpu()
                for name, value in self._shadow.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        version = state_dict.get("ema_state_version")
        if version != EMA_STATE_VERSION:
            raise RuntimeError(
                f"Unsupported EMA checkpoint state version {version!r}."
            )
        checkpoint_decay = float(state_dict["decay"])
        checkpoint_every = int(state_dict["update_every_n_steps"])
        if checkpoint_decay != self.decay:
            raise RuntimeError(
                f"EMA decay changed across resume: {checkpoint_decay} -> {self.decay}."
            )
        if checkpoint_every != self.update_every_n_steps:
            raise RuntimeError(
                "EMA update frequency changed across resume: "
                f"{checkpoint_every} -> {self.update_every_n_steps}."
            )
        self.num_updates = int(state_dict["num_updates"])
        self._shadow = {
            name: value.detach().clone()
            for name, value in state_dict["shadow"].items()
        }


def _ema_state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    matches = [
        state
        for state in checkpoint.get("callbacks", {}).values()
        if isinstance(state, dict)
        and state.get("ema_state_version") == EMA_STATE_VERSION
        and "shadow" in state
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one EMA callback state, found {len(matches)}."
        )
    return matches[0]


def export_ema_checkpoint(source: Path, destination: Path) -> Path:
    """Export an evaluation-only checkpoint whose state_dict contains EMA weights."""

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise RuntimeError(f"{source} does not contain a Lightning state_dict.")
    ema_state = _ema_state_from_checkpoint(checkpoint)
    shadow = ema_state["shadow"]
    raw_state = checkpoint["state_dict"]
    missing = sorted(set(shadow) - set(raw_state))
    if missing:
        raise RuntimeError(f"EMA parameters missing from model state_dict: {missing}")

    ema_model_state = dict(raw_state)
    ema_model_state.update(
        {name: value.detach().cpu().clone() for name, value in shadow.items()}
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": ema_model_state,
            "ema_metadata": {
                "source_checkpoint": str(source),
                "decay": float(ema_state["decay"]),
                "num_updates": int(ema_state["num_updates"]),
                "evaluation_only": True,
            },
        },
        destination,
    )
    return destination
