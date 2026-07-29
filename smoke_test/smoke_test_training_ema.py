"""Smoke tests for optional, checkpoint-safe training EMA."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "server_example"))

from training_ema import TrainingEMA, export_ema_checkpoint


class TinyRegression(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.linear(x)

    def _loss(self, batch):
        x, target = batch
        return torch.nn.functional.mse_loss(self(x), target)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("loss_epoch", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("val/loss_epoch", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05)


class TinyDualRegression(TinyRegression):
    def __init__(self, ema: TrainingEMA) -> None:
        super().__init__()
        self.ema = ema

    def validation_step(self, batch, batch_idx):
        x, target = batch
        raw_prediction = self(x)
        raw_loss = torch.nn.functional.mse_loss(raw_prediction, target)
        raw_weight = self.linear.weight.detach().clone()
        ema_prediction = functional_call(
            self.linear,
            self.ema.functional_parameters_for(self.linear, prefix="linear."),
            (x,),
        )
        ema_loss = torch.nn.functional.mse_loss(ema_prediction, target)
        torch.testing.assert_close(self.linear.weight, raw_weight)
        self.log(
            "val/raw_loss_epoch",
            raw_loss,
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
        )
        self.log(
            "val/ema_loss_epoch",
            ema_loss,
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
        )


def loader() -> DataLoader:
    x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]] * 2
    )
    target = (2.0 * x[:, :1]) - x[:, 1:]
    return DataLoader(TensorDataset(x, target), batch_size=2, shuffle=False)


def trainer(
    root: Path,
    ema: TrainingEMA,
    checkpoint: ModelCheckpoint,
    max_epochs: int,
) -> L.Trainer:
    return L.Trainer(
        default_root_dir=root,
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        accumulate_grad_batches=2,
        callbacks=[ema, checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )


def test_manual_update_and_swap() -> None:
    module = TinyRegression()
    with torch.no_grad():
        module.linear.weight.fill_(1.0)
    callback = TrainingEMA(decay=0.9, validate_with_ema=True)
    fake_trainer = type("Trainer", (), {"global_step": 0})()
    callback.on_fit_start(fake_trainer, module)

    with torch.no_grad():
        module.linear.weight.fill_(3.0)
    fake_trainer.global_step = 1
    callback.on_train_batch_end(fake_trainer, module, None, None, 0)
    expected_ema = torch.full_like(module.linear.weight, 1.2)
    torch.testing.assert_close(callback._shadow["linear.weight"], expected_ema)

    callback.on_validation_start(fake_trainer, module)
    torch.testing.assert_close(module.linear.weight, expected_ema)
    callback.on_validation_end(fake_trainer, module)
    torch.testing.assert_close(
        module.linear.weight,
        torch.full_like(module.linear.weight, 3.0),
    )


def test_lightning_checkpoint_resume_and_export() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint_dir = root / "checkpoints"
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch-{epoch:03d}",
            monitor="val/loss_epoch",
            mode="min",
            save_last=True,
            save_top_k=1,
        )
        ema = TrainingEMA(decay=0.9, validate_with_ema=True)
        first_trainer = trainer(root, ema, checkpoint, max_epochs=2)
        first_trainer.fit(
            TinyRegression(),
            train_dataloaders=loader(),
            val_dataloaders=loader(),
        )
        assert ema.num_updates == first_trainer.global_step
        assert checkpoint.last_model_path

        saved = torch.load(
            checkpoint.last_model_path,
            map_location="cpu",
            weights_only=False,
        )
        assert "TrainingEMA" in saved["callbacks"]
        saved_updates = saved["callbacks"]["TrainingEMA"]["num_updates"]
        assert saved_updates == ema.num_updates

        resumed_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="resumed-{epoch:03d}",
            monitor="val/loss_epoch",
            mode="min",
            save_last=True,
            save_top_k=1,
        )
        resumed_ema = TrainingEMA(decay=0.9, validate_with_ema=True)
        resumed_trainer = trainer(
            root,
            resumed_ema,
            resumed_checkpoint,
            max_epochs=3,
        )
        resumed_trainer.fit(
            TinyRegression(),
            train_dataloaders=loader(),
            val_dataloaders=loader(),
            ckpt_path=checkpoint.last_model_path,
        )
        assert resumed_ema.num_updates > saved_updates

        destination = export_ema_checkpoint(
            Path(resumed_checkpoint.last_model_path),
            checkpoint_dir / "ema-last.ckpt",
        )
        exported = torch.load(destination, map_location="cpu", weights_only=False)
        assert set(exported) == {"state_dict", "ema_metadata"}
        assert exported["ema_metadata"]["evaluation_only"] is True
        callback_state = torch.load(
            resumed_checkpoint.last_model_path,
            map_location="cpu",
            weights_only=False,
        )["callbacks"]["TrainingEMA"]
        torch.testing.assert_close(
            exported["state_dict"]["linear.weight"],
            callback_state["shadow"]["linear.weight"],
        )


def test_dual_raw_and_ema_checkpoints() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint_dir = root / "checkpoints"
        raw_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="raw-epoch-{epoch:03d}",
            monitor="val/raw_loss_epoch",
            mode="min",
            save_last=True,
            save_top_k=1,
        )
        ema_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="ema-source-epoch-{epoch:03d}",
            monitor="val/ema_loss_epoch",
            mode="min",
            save_last=False,
            save_top_k=1,
        )
        ema = TrainingEMA(decay=0.9, validate_with_ema=False)
        dual_trainer = L.Trainer(
            default_root_dir=root,
            max_epochs=3,
            accelerator="cpu",
            devices=1,
            precision="32-true",
            callbacks=[ema, raw_checkpoint, ema_checkpoint],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
        )
        module = TinyDualRegression(ema)
        dual_trainer.fit(
            module,
            train_dataloaders=loader(),
            val_dataloaders=loader(),
        )

        assert raw_checkpoint.best_model_path
        assert ema_checkpoint.best_model_path
        assert Path(raw_checkpoint.best_model_path).is_file()
        assert Path(ema_checkpoint.best_model_path).is_file()
        assert raw_checkpoint.best_model_score is not None
        assert ema_checkpoint.best_model_score is not None
        exported = export_ema_checkpoint(
            Path(ema_checkpoint.best_model_path),
            checkpoint_dir / "ema-best.ckpt",
        )
        assert exported.is_file()


def main() -> None:
    test_manual_update_and_swap()
    test_lightning_checkpoint_resume_and_export()
    test_dual_raw_and_ema_checkpoints()
    print("training EMA smoke tests passed")


if __name__ == "__main__":
    main()
