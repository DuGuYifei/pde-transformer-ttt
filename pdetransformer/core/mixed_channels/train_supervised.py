from pathlib import Path
from typing import List, Union

from torch.nn.functional import mse_loss

from ...utils import instantiate_from_config
import torch.nn as nn
import lightning
import torch
from tqdm import tqdm
import numpy as np

class SingleStepSupervised(lightning.LightningModule):
    def __init__(self,
                 model: Union[dict, nn.Module],
                 ckpt_path=None,
                 ignore_keys=None,
                 image_key=0,
                 monitor=None,
                 detect_zero_grad=False,
                 normalize_channels=False,
                 optimizer='adamw',
                 use_ttt_state_cache_inference: bool = False,
                 use_ttt_state_cache_train: bool = False,
                 ):
        super(SingleStepSupervised, self).__init__()

        self.image_key = image_key
        self.optimizer = optimizer
        self.normalize_channels = normalize_channels
        self.use_ttt_state_cache_inference = use_ttt_state_cache_inference
        self.use_ttt_state_cache_train = use_ttt_state_cache_train

        if isinstance(model, dict):
            self.model: nn.Module = instantiate_from_config(model)
        else:
            self.model = model

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.downsample_factor = 1
        self.detect_zero_grad = detect_zero_grad

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                conditioning: torch.Tensor = None) -> torch.Tensor:

        return self.model(x, t)

    def get_pipeline_args(self):
        return {
            "unet": self.model,
        }

    def init_from_ckpt(self, path: str, ignore_keys:List[str]=None):
        if ignore_keys is None:
            ignore_keys = list()

        if Path(path).is_dir():
            path = Path(path).joinpath("last.ckpt")
        else:
            path = Path(path)

        if path.is_file():
            sd = torch.load(path, map_location="cpu")["state_dict"]
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
            self.load_state_dict(sd, strict=False)
            print(f"Restored from {path}")

    def get_input(self, batch, batch_dim=True, trim:int=0):

        data: torch.Tensor = batch["data"]
        meta_data_loading: dict = batch["loading_metadata"]
        meta_data_physical: dict = batch["physical_metadata"]

        if batch_dim:
            x: torch.Tensor = data[:, 0+trim]
            y: torch.Tensor = data[:, 1+trim:]

            task_idx = meta_data_physical['PDE'][:, 0]

        else:
            x: torch.Tensor = data[0+trim]
            y: torch.Tensor = data[1+trim:]
            task_idx = meta_data_physical['PDE']

            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(y, 0)

            if not torch.is_tensor(task_idx):
                task_idx = torch.tensor(task_idx)

        if self.downsample_factor > 1:
            # downsample with average pooling
            x = nn.functional.avg_pool2d(x, self.downsample_factor)

            num_batches = y.shape[0]
            y = y.reshape(-1, y.shape[-3], y.shape[-2], y.shape[-1])
            y = nn.functional.avg_pool2d(y, self.downsample_factor)
            y = y.reshape(num_batches, -1, y.shape[-3], y.shape[-2], y.shape[-1])

        return x, y, task_idx

    def test_step(self, batch, batch_idx):

        return 0, {}


    def training_step(self, batch, batch_idx):

        input, target, labels = self.get_input(batch)

        model_output = self.model(
            input,
            class_labels=labels,
            ttt_state_cache={} if self.use_ttt_state_cache_train else None,
            return_ttt_state_cache=self.use_ttt_state_cache_train,
        )
        pred = model_output.sample

        # select last frame as target
        target = target[:, -1]

        if self.normalize_channels:
            # normalize channels (mean, std)
            target = ((target - target.mean(dim=(2, 3), keepdim=True))
                      / (target.std(dim=(2, 3), keepdim=True) + 1e-4))
            pred = ((pred - target.mean(dim=(2, 3), keepdim=True))
                    / (target.std(dim=(2, 3), keepdim=True)+ 1e-4))

        loss = mse_loss(pred, target)

        self.log("loss", loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        input, target, labels = self.get_input(batch)

        model_output = self.model(
            input,
            class_labels=labels,
            ttt_state_cache={} if self.use_ttt_state_cache_train else None,
            return_ttt_state_cache=self.use_ttt_state_cache_train,
        )
        pred = model_output.sample

        # select last frame as target
        target = target[:, -1]

        if self.normalize_channels:
            # normalize channels (mean, std)
            target = ((target - target.mean(dim=(2, 3), keepdim=True))
                      / (target.std(dim=(2, 3), keepdim=True) + 1e-4))
            pred = ((pred - pred.mean(dim=(2, 3), keepdim=True))
                    / (pred.std(dim=(2, 3), keepdim=True)+ 1e-4))

        loss = mse_loss(pred, target)

        self.log("val/loss", loss.item(), prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return {"val/loss": loss}

    def on_after_backward(self):
        if self.detect_zero_grad:
            for name, param in self.named_parameters():
                if param.grad is None:
                    print('Detected params without gradient: ', name)

    def configure_optimizers(self):

        if self.optimizer == 'adamw':

            opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                    weight_decay=1e-15)

        elif self.optimizer == 'adabelief':

            from adabelief_pytorch import AdaBelief # noqa
            opt = AdaBelief(self.parameters(), lr=self.learning_rate)

        else:

            raise ValueError(f"Optimizer {self.optimizer} not supported;")

        return [opt]

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def predict_step(self, previous_frame,
                     num_inference_steps=100,
                     generator=None,
                     ttt_state_cache=None,
                     return_ttt_state_cache: bool = False,
                     **kwargs):

        model_output = self.model(
            previous_frame,
            ttt_state_cache=ttt_state_cache,
            return_ttt_state_cache=return_ttt_state_cache,
            **kwargs,
        )
        pred = model_output.sample
        if return_ttt_state_cache:
            return pred, model_output.ttt_state_cache
        return pred

    def postprocess_output(self, x0, physical_metadata):

        fields = physical_metadata['Fields'].to(x0.device)
        fields = fields > 0 # if field id > 0, then it corresponds to a physical field, padded otherwise
        # set padded values to zero

        if len(fields.shape) == 1:
            fields = fields.unsqueeze(0)

        x0 = torch.einsum('bc..., bc -> bc...', x0, fields)

        return x0

    def predict_raw(self, input_0,
                    input_1,
                    labels,
                    device,
                    num_frames=20,
                    generator=None,):

        num_inference_steps = 100

        with torch.no_grad():

            input_0 = input_0.to(device)
            input_1 = input_1.to(device)[:, :num_frames - 1]
            labels = labels.to(device)

            if generator is None:
                generator = (torch.Generator(device=input_0.device)
                             .manual_seed(2024))

            frames = [input_0.cpu()]
            previous_frame = input_0
            ttt_state_cache = {} if self.use_ttt_state_cache_inference else None

            for i in tqdm(range(num_frames - 1)):

                if self.use_ttt_state_cache_inference:
                    x0, ttt_state_cache = self.predict_step(
                        previous_frame, generator=generator, class_labels=labels,
                        num_inference_steps=num_inference_steps,
                        ttt_state_cache=ttt_state_cache,
                        return_ttt_state_cache=True,
                    )
                else:
                    x0 = self.predict_step(previous_frame, generator=generator, class_labels=labels,
                                           num_inference_steps=num_inference_steps)

                previous_frame = x0
                frames.append(x0.cpu())

            vid = np.array([frame.numpy() for frame in frames])
            vid = np.swapaxes(vid, 0, 1)

            reference = np.array(torch.concat([frames[0].unsqueeze(1),
                                               input_1.cpu()], dim=1))

        return vid, reference


    def predict(self, batch, device, num_frames=20, generator=None,
                output_type='numpy', num_inference_steps=100, return_dict=True,
                reference_boundary=False, batch_dim=True, trim:int=0):

        boundary_slice = 0

        with torch.no_grad():

            input_0, input_1, labels = self.get_input(batch, batch_dim=batch_dim, trim=trim)

            input_0 = input_0.to(device)
            input_1 = input_1.to(device)[:, :num_frames-1]
            labels = labels.to(device)

            if generator is None:
                generator = (torch.Generator(device=input_0.device)
                             .manual_seed(2024))

            frames = [input_0.cpu()]
            previous_frame = input_0
            ttt_state_cache = {} if self.use_ttt_state_cache_inference else None

            for i in tqdm(range(num_frames-1)):

                if self.use_ttt_state_cache_inference:
                    x0, ttt_state_cache = self.predict_step(
                        previous_frame, generator=generator, class_labels=labels,
                        num_inference_steps=num_inference_steps,
                        ttt_state_cache=ttt_state_cache,
                        return_ttt_state_cache=True,
                    )
                else:
                    x0 = self.predict_step(previous_frame, generator=generator, class_labels=labels,
                                           num_inference_steps=num_inference_steps)
                x0 = self.postprocess_output(x0, batch['physical_metadata'])

                # fill boundaries with reference
                if reference_boundary:

                    if len(x0.shape) == 4: # 2D
                        x0[:, :, 0:boundary_slice, :] = input_1[:, i, :, 0:boundary_slice, :]
                        x0[:, :, -boundary_slice:, :] = input_1[:, i, :, -boundary_slice:, :]
                        x0[:, :, :, 0:boundary_slice] = input_1[:, i, :, :, 0:boundary_slice]
                        x0[:, :, :, -boundary_slice:] = input_1[:, i, :, :, -boundary_slice:]

                    if len(x0.shape) == 5: # 3D
                        x0[:, :, 0:boundary_slice, :, :] = input_1[:, i, :, 0:boundary_slice, :, :]
                        x0[:, :, -boundary_slice:, :, :] = input_1[:, i, :, -boundary_slice:, :, :]
                        x0[:, :, :, 0:boundary_slice, :] = input_1[:, i, :, :, 0:boundary_slice, :]
                        x0[:, :, :, -boundary_slice:, :] = input_1[:, i, :, :, -boundary_slice:, :]
                        x0[:, :, :, :, 0:boundary_slice] = input_1[:, i, :, :, :, 0:boundary_slice]
                        x0[:, :, :, :, -boundary_slice:] = input_1[:, i, :, :, :, -boundary_slice:]

                previous_frame = x0
                frames.append(x0.cpu())

            vid = np.array([frame.numpy() for frame in frames])
            vid = np.swapaxes(vid, 0, 1)

            reference = np.array(torch.concat([frames[0].unsqueeze(1),
                                               input_1.cpu()], dim=1))

            if not batch_dim:
                vid = vid[0]
                reference = reference[0]

        return vid, reference


def _detach_state_tree(state):
    if state is None:
        return None
    if torch.is_tensor(state):
        return state.detach()
    if isinstance(state, dict):
        return {key: _detach_state_tree(value) for key, value in state.items()}
    if isinstance(state, tuple):
        return tuple(_detach_state_tree(value) for value in state)
    if isinstance(state, list):
        return [_detach_state_tree(value) for value in state]
    return state


class TemporalRolloutSupervised(SingleStepSupervised):
    """Autoregressive PDE training with a persistent TTT state and TBPTT."""

    def __init__(
        self,
        *args,
        train_unrolling_steps: int = 29,
        tbptt_chunk_size: int = 4,
        gradient_accumulation_batches: int = 1,
        temporal_ttt_learning_rate: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            use_ttt_state_cache_inference=True,
            use_ttt_state_cache_train=True,
            **kwargs,
        )
        if train_unrolling_steps < 1:
            raise ValueError("train_unrolling_steps must be positive")
        if tbptt_chunk_size < 1:
            raise ValueError("tbptt_chunk_size must be positive")
        if gradient_accumulation_batches < 1:
            raise ValueError("gradient_accumulation_batches must be positive")

        self.train_unrolling_steps = train_unrolling_steps
        self.tbptt_chunk_size = tbptt_chunk_size
        self.gradient_accumulation_batches = gradient_accumulation_batches
        self.temporal_ttt_learning_rate = temporal_ttt_learning_rate
        self.automatic_optimization = False
        self._batches_since_optimizer_step = 0

    def configure_optimizers(self):
        temporal_params = []
        base_params = []
        for name, parameter in self.named_parameters():
            if ".temporal_ttt." in name:
                temporal_params.append(parameter)
            else:
                base_params.append(parameter)
        if not temporal_params:
            raise RuntimeError("TemporalRolloutSupervised requires a temporal_ttt module")

        temporal_lr = self.temporal_ttt_learning_rate or self.learning_rate
        return torch.optim.AdamW(
            [
                {"params": base_params, "lr": self.learning_rate},
                {"params": temporal_params, "lr": temporal_lr},
            ],
            weight_decay=1e-15,
        )

    def _validate_sequence_length(self, target: torch.Tensor) -> int:
        available_steps = target.shape[1]
        if available_steps < self.train_unrolling_steps:
            raise ValueError(
                f"batch contains {available_steps} target steps, but temporal training "
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
        optimizer.step()
        optimizer.zero_grad()
        self._batches_since_optimizer_step = 0

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
            chunk_loss = torch.zeros((), device=input_frame.device)
            for step in range(chunk_start, chunk_end):
                model_output = self.model(
                    previous_frame,
                    class_labels=labels,
                    ttt_state_cache=state,
                    return_ttt_state_cache=True,
                )
                pred = model_output.sample
                pred_for_loss, target_for_loss = self._normalize_pair(pred, targets[:, step])
                chunk_loss = chunk_loss + mse_loss(pred_for_loss, target_for_loss)
                previous_frame = pred.detach()
                state = model_output.ttt_state_cache

            scaled_chunk_loss = chunk_loss / (
                num_steps * self.gradient_accumulation_batches
            )
            self.manual_backward(scaled_chunk_loss)
            total_loss = total_loss + chunk_loss.detach()
            state = _detach_state_tree(state)

        self._batches_since_optimizer_step += 1
        self._optimizer_step_if_ready(optimizer)

        mean_loss = total_loss / num_steps
        self.log(
            "loss",
            mean_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=input_frame.shape[0],
        )
        return mean_loss

    def on_train_epoch_end(self) -> None:
        if self._batches_since_optimizer_step:
            self._optimizer_step_if_ready(self.optimizers(), force=True)

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
            "val/loss",
            mean_loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=input_frame.shape[0],
        )
        return {"val/loss": mean_loss}
