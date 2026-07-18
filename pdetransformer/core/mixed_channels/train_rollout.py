"""Autoregressive rollout objective for stateless PDE backbones."""

import torch
from torch.nn.functional import mse_loss

from .train_supervised import SingleStepSupervised


class AutoregressiveRolloutSupervised(SingleStepSupervised):
    """Train a stateless PDE model on an autoregressive rollout with TBPTT."""

    def __init__(
        self,
        *args,
        train_unrolling_steps: int = 29,
        tbptt_chunk_size: int = 4,
        gradient_accumulation_batches: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if train_unrolling_steps < 1:
            raise ValueError("train_unrolling_steps must be positive")
        if tbptt_chunk_size < 1:
            raise ValueError("tbptt_chunk_size must be positive")
        if gradient_accumulation_batches < 1:
            raise ValueError("gradient_accumulation_batches must be positive")

        self.train_unrolling_steps = train_unrolling_steps
        self.tbptt_chunk_size = tbptt_chunk_size
        self.gradient_accumulation_batches = gradient_accumulation_batches
        self.automatic_optimization = False
        self._batches_since_optimizer_step = 0

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("AutoregressiveRolloutSupervised has no trainable parameters")
        return torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=1e-15)

    def _validate_sequence_length(self, target: torch.Tensor) -> int:
        available_steps = target.shape[1]
        if available_steps < self.train_unrolling_steps:
            raise ValueError(
                f"batch contains {available_steps} target steps, but rollout training "
                f"requires {self.train_unrolling_steps}"
            )
        return self.train_unrolling_steps

    def _normalize_pair(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.normalize_channels:
            return pred, target
        mean = target.mean(dim=(2, 3), keepdim=True)
        std = target.std(dim=(2, 3), keepdim=True) + 1e-4
        return (pred - mean) / std, (target - mean) / std

    def _optimizer_step_if_ready(self, optimizer, force: bool = False) -> None:
        if self._batches_since_optimizer_step == 0:
            return
        if not force and self._batches_since_optimizer_step < self.gradient_accumulation_batches:
            return
        if force and self._batches_since_optimizer_step < self.gradient_accumulation_batches:
            scale = self.gradient_accumulation_batches / self._batches_since_optimizer_step
            for parameter in self.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        optimizer.step()
        optimizer.zero_grad()
        self._batches_since_optimizer_step = 0

    def _rollout_chunk(
        self,
        previous_frame: torch.Tensor,
        targets: torch.Tensor,
        labels: torch.Tensor,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_loss = torch.zeros((), device=previous_frame.device)
        for step in range(start, end):
            pred = self.model(previous_frame, class_labels=labels).sample
            pred_for_loss, target_for_loss = self._normalize_pair(pred, targets[:, step])
            chunk_loss = chunk_loss + mse_loss(pred_for_loss, target_for_loss)
            # Keep the prediction graph inside this chunk. Only the caller detaches
            # at a TBPTT boundary.
            previous_frame = pred
        return previous_frame, chunk_loss

    def training_step(self, batch, batch_idx):
        input_frame, targets, labels = self.get_input(batch)
        num_steps = self._validate_sequence_length(targets)
        optimizer = self.optimizers()
        if self._batches_since_optimizer_step == 0:
            optimizer.zero_grad()

        previous_frame = input_frame
        total_loss = torch.zeros((), device=input_frame.device)
        for chunk_start in range(0, num_steps, self.tbptt_chunk_size):
            chunk_end = min(chunk_start + self.tbptt_chunk_size, num_steps)
            previous_frame, chunk_loss = self._rollout_chunk(
                previous_frame,
                targets,
                labels,
                chunk_start,
                chunk_end,
            )
            scaled_chunk_loss = chunk_loss / (
                num_steps * self.gradient_accumulation_batches
            )
            self.manual_backward(scaled_chunk_loss)
            total_loss = total_loss + chunk_loss.detach()
            previous_frame = previous_frame.detach()

        self._batches_since_optimizer_step += 1
        self._optimizer_step_if_ready(optimizer)

        mean_loss = total_loss / num_steps
        self.log(
            "loss", mean_loss, prog_bar=True, logger=True, on_step=True,
            on_epoch=True, sync_dist=True, batch_size=input_frame.shape[0],
        )
        return mean_loss

    def on_train_epoch_end(self) -> None:
        if self._batches_since_optimizer_step:
            self._optimizer_step_if_ready(self.optimizers(), force=True)

    def validation_step(self, batch, batch_idx):
        input_frame, targets, labels = self.get_input(batch)
        num_steps = self._validate_sequence_length(targets)
        previous_frame = input_frame
        total_loss = torch.zeros((), device=input_frame.device)

        for step in range(num_steps):
            pred = self.model(previous_frame, class_labels=labels).sample
            pred_for_loss, target_for_loss = self._normalize_pair(pred, targets[:, step])
            total_loss = total_loss + mse_loss(pred_for_loss, target_for_loss)
            previous_frame = pred

        mean_loss = total_loss / num_steps
        self.log(
            "val/loss", mean_loss, prog_bar=True, logger=True, on_step=False,
            on_epoch=True, sync_dist=True, batch_size=input_frame.shape[0],
        )
        return {"val/loss": mean_loss}


def detach_state_tree(state):
    if state is None:
        return None
    if torch.is_tensor(state):
        return state.detach()
    if isinstance(state, dict):
        return {key: detach_state_tree(value) for key, value in state.items()}
    if isinstance(state, tuple):
        return tuple(detach_state_tree(value) for value in state)
    if isinstance(state, list):
        return [detach_state_tree(value) for value in state]
    return state


class PersistentAutoregressiveRolloutSupervised(AutoregressiveRolloutSupervised):
    """Rollout training with each spatial Linear TTT state carried across steps."""

    def _persistent_rollout_chunk(
        self,
        previous_frame: torch.Tensor,
        targets: torch.Tensor,
        labels: torch.Tensor,
        state: dict,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        chunk_loss = torch.zeros((), device=previous_frame.device)
        for step in range(start, end):
            model_output = self.model(
                previous_frame,
                class_labels=labels,
                ttt_state_cache=state,
                return_ttt_state_cache=True,
            )
            pred = model_output.sample
            pred_for_loss, target_for_loss = self._normalize_pair(pred, targets[:, step])
            chunk_loss = chunk_loss + mse_loss(pred_for_loss, target_for_loss)
            previous_frame = pred
            state = model_output.ttt_state_cache
        return previous_frame, chunk_loss, state

    def training_step(self, batch, batch_idx):
        input_frame, targets, labels = self.get_input(batch)
        num_steps = self._validate_sequence_length(targets)
        optimizer = self.optimizers()
        if self._batches_since_optimizer_step == 0:
            optimizer.zero_grad()

        previous_frame = input_frame
        state = {}
        total_loss = torch.zeros((), device=input_frame.device)
        for chunk_start in range(0, num_steps, self.tbptt_chunk_size):
            chunk_end = min(chunk_start + self.tbptt_chunk_size, num_steps)
            previous_frame, chunk_loss, state = self._persistent_rollout_chunk(
                previous_frame,
                targets,
                labels,
                state,
                chunk_start,
                chunk_end,
            )
            scaled_chunk_loss = chunk_loss / (
                num_steps * self.gradient_accumulation_batches
            )
            self.manual_backward(scaled_chunk_loss)
            total_loss = total_loss + chunk_loss.detach()
            previous_frame = previous_frame.detach()
            state = detach_state_tree(state)

        self._batches_since_optimizer_step += 1
        self._optimizer_step_if_ready(optimizer)

        mean_loss = total_loss / num_steps
        self.log(
            "loss", mean_loss, prog_bar=True, logger=True, on_step=True,
            on_epoch=True, sync_dist=True, batch_size=input_frame.shape[0],
        )
        return mean_loss

    def validation_step(self, batch, batch_idx):
        input_frame, targets, labels = self.get_input(batch)
        num_steps = self._validate_sequence_length(targets)
        previous_frame = input_frame
        state = {}
        total_loss = torch.zeros((), device=input_frame.device)

        for step in range(num_steps):
            model_output = self.model(
                previous_frame,
                class_labels=labels,
                ttt_state_cache=state,
                return_ttt_state_cache=True,
            )
            pred = model_output.sample
            pred_for_loss, target_for_loss = self._normalize_pair(pred, targets[:, step])
            total_loss = total_loss + mse_loss(pred_for_loss, target_for_loss)
            previous_frame = pred
            state = model_output.ttt_state_cache

        mean_loss = total_loss / num_steps
        self.log(
            "val/loss", mean_loss, prog_bar=True, logger=True, on_step=False,
            on_epoch=True, sync_dist=True, batch_size=input_frame.shape[0],
        )
        return {"val/loss": mean_loss}
