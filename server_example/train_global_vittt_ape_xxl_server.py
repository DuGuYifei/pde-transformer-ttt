"""Focused server entrypoint for fair Attention/ViTTT-PDE comparisons."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import types
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from torch.func import functional_call
from torch.nn.functional import mse_loss

# Prefer the reviewed source tree over an older installed wheel on the server.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for package_name, package_path in {
    "pdetransformer": REPO_ROOT / "pdetransformer",
    "pdetransformer.core": REPO_ROOT / "pdetransformer" / "core",
}.items():
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package

from pdetransformer.core.mixed_channels import PDETransformer, SingleStepSupervised
from pdetransformer.data.pbdl_datatypes.ape_2d_splits import (
    DATASET_PROFILES,
    SEPARATE_TEST_DATASETS,
    ape_2d_xxl_simulation_split,
    dataset_names_for_profile,
)
from training_ema import TrainingEMA, export_ema_checkpoint


REQUIRED_CONFIG_KEYS = {
    "data_dir",
    "run_root",
    "run_name",
    "model_type",
    "in_channels",
    "out_channels",
    "patch_size",
    "periodic",
    "carrier_token_active",
    "token_mixer_type",
    "vittt_inner_lr",
    "vittt_head_dim",
    "learning_rate",
    "max_epochs",
    "batch_size",
    "num_workers",
    "devices",
    "strategy",
    "precision",
    "accumulate_grad_batches",
    "seed",
    "downsample_factor",
    "sample_size",
    "train_unrolling_steps",
    "test_unrolling_steps",
    "max_channels",
    "auto_resume",
    "dataset_profile",
}
OPTIONAL_CONFIG_DEFAULTS = {
    "init_checkpoint": None,
    "window_ttt_update_mode": "full_batch",
    "window_ttt_chunk_size": 16,
    "ttt_layer_type": "linear",
    "ttt_mini_batch_size": 16,
    "ttt_base_lr": 1.0,
    "ttt_use_gate": False,
    "ttt_scan_checkpoint_group_size": 0,
    "vittt_padding_mode": "zero",
    "use_ema": False,
    "ema_decay": 0.999,
    "ema_update_every_n_steps": 1,
    "ema_validate": True,
    "ema_dual_validation": False,
}
OPTIONAL_CONFIG_KEYS = set(OPTIONAL_CONFIG_DEFAULTS)


class EpochSummary(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.started_at = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = (time.time() - self.started_at) / 60.0
        loss = trainer.callback_metrics.get("loss_epoch")
        loss_text = f" loss={float(loss):.6g}" if loss is not None else ""
        print(
            f"[train] epoch={trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"step={trainer.global_step} elapsed={elapsed:.1f}min{loss_text}"
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        raw_loss = trainer.callback_metrics.get("val/raw_loss_epoch")
        ema_loss = trainer.callback_metrics.get("val/ema_loss_epoch")
        if raw_loss is not None and ema_loss is not None:
            print(
                f"[val] epoch={trainer.current_epoch + 1} "
                f"raw_loss={float(raw_loss):.6g} ema_loss={float(ema_loss):.6g}"
            )
            return
        loss = trainer.callback_metrics.get("val/loss_epoch")
        if loss is not None:
            print(f"[val] epoch={trainer.current_epoch + 1} loss={float(loss):.6g}")


class DualEMAValidationSupervised(SingleStepSupervised):
    """Evaluate raw and EMA weights on the same validation batches."""

    ema_callback: TrainingEMA | None = None

    def attach_ema_callback(self, callback: TrainingEMA) -> None:
        self.ema_callback = callback

    def _validation_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.normalize_channels:
            target = (
                target - target.mean(dim=(2, 3), keepdim=True)
            ) / (target.std(dim=(2, 3), keepdim=True) + 1e-4)
            prediction = (
                prediction - prediction.mean(dim=(2, 3), keepdim=True)
            ) / (prediction.std(dim=(2, 3), keepdim=True) + 1e-4)
        return mse_loss(prediction, target)

    def validation_step(self, batch, batch_idx):
        if self.ema_callback is None:
            raise RuntimeError("Dual EMA validation requires an attached TrainingEMA callback.")

        inputs, targets, labels = self.get_input(batch)
        targets = targets[:, -1]
        raw_prediction = self.model(inputs, class_labels=labels).sample
        raw_loss = self._validation_loss(raw_prediction, targets)

        ema_parameters = self.ema_callback.functional_parameters_for(
            self.model,
            prefix="model.",
        )
        ema_prediction = functional_call(
            self.model,
            ema_parameters,
            (inputs,),
            {"class_labels": labels},
        ).sample
        ema_loss = self._validation_loss(ema_prediction, targets)

        batch_size = int(inputs.shape[0])
        self.log(
            "val/raw_loss_epoch",
            raw_loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val/ema_loss_epoch",
            ema_loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return {
            "val/raw_loss_epoch": raw_loss,
            "val/ema_loss_epoch": ema_loss,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--strategy", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulate-grad-batches", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the experiment seed for independent repeated runs.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Override the output root, primarily for isolated server smoke tests.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Override the run name, primarily for isolated server smoke tests.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        help="Limit training batches for a server memory smoke test.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        help="Limit validation batches for a server memory smoke test.",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing last checkpoint.")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Build the model and print its parameter count without loading data.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Validate required HDF5 files and disjoint simulation splits, then exit.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    unknown = sorted(set(config) - REQUIRED_CONFIG_KEYS - OPTIONAL_CONFIG_KEYS)
    if missing or unknown:
        raise ValueError(f"Invalid config: missing={missing}, unknown={unknown}")
    for key, value in OPTIONAL_CONFIG_DEFAULTS.items():
        config.setdefault(key, value)
    if config["token_mixer_type"] not in {
        "attention",
        "ttt_sequence",
        "vittt",
        "window_linear_ttt",
        "window_fullbatch_mlp_ttt",
    }:
        raise ValueError(f"Unsupported token_mixer_type={config['token_mixer_type']!r}")
    if config["carrier_token_active"] and config["token_mixer_type"] != "attention":
        raise ValueError("TTT token mixers cannot enable carrier tokens.")
    if config["window_ttt_update_mode"] not in {
        "full_batch",
        "token_sequential",
        "window_sequential",
    }:
        raise ValueError(
            "window_ttt_update_mode must be full_batch, token_sequential, "
            "or window_sequential."
        )
    if int(config["window_ttt_chunk_size"]) < 1:
        raise ValueError("window_ttt_chunk_size must be positive.")
    if config["dataset_profile"] not in DATASET_PROFILES:
        raise ValueError(
            f"Unsupported dataset_profile={config['dataset_profile']!r}; "
            f"expected one of {sorted(DATASET_PROFILES)}."
        )
    if not 0.0 <= float(config["ema_decay"]) < 1.0:
        raise ValueError("ema_decay must be in [0, 1).")
    if int(config["ema_update_every_n_steps"]) < 1:
        raise ValueError("ema_update_every_n_steps must be positive.")
    if config["ema_dual_validation"] and not config["use_ema"]:
        raise ValueError("ema_dual_validation requires use_ema=true.")
    if config["ema_dual_validation"] and not config["ema_validate"]:
        raise ValueError("ema_dual_validation requires ema_validate=true.")
    return config


def build_data_module(config: dict):
    # Delay this import: the bundled PBDL loader refreshes global_index.json as
    # an import side effect, which is unnecessary for --help/config review.
    from pdetransformer.data import MultiDataModule

    return MultiDataModule(
        path_index={"2D_APE_xxl": str(Path(config["data_dir"]).expanduser())},
        dataset_names=list(dataset_names_for_profile(config["dataset_profile"])),
        dataset_type="2D_APE_xxl",
        unrolling_steps=config["train_unrolling_steps"],
        test_unrolling_steps=config["test_unrolling_steps"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        cache_strategy="none",
        different_resolution_strategy="none",
        normalize_data="mean-std",
        normalize_const="mean-std",
        downsample_factor=config["downsample_factor"],
        max_channels=config["max_channels"],
        dataset_profile=config["dataset_profile"],
    )


def validate_dataset_files(config: dict) -> None:
    """Fail before PBDL can fall back to a different remote dataset release."""

    import h5py

    data_dir = Path(config["data_dir"]).expanduser().resolve()
    dataset_names = dataset_names_for_profile(config["dataset_profile"])
    rows = []
    for name in dataset_names:
        train_path = data_dir / f"{name}.hdf5"
        if not train_path.is_file():
            raise FileNotFoundError(f"Required training dataset is missing: {train_path}")
        with h5py.File(train_path, "r") as handle:
            train_file_sims = len(handle["sims"])

        train_sims, test_sims = ape_2d_xxl_simulation_split(
            name,
            config["dataset_profile"],
        )
        if name in SEPARATE_TEST_DATASETS:
            test_path = data_dir / f"{name}_test.hdf5"
            if not test_path.is_file():
                raise FileNotFoundError(f"Required test dataset is missing: {test_path}")
            with h5py.File(test_path, "r") as handle:
                test_file_sims = len(handle["sims"])
            rows.append(
                f"{name}: train_file={train_file_sims} test_file={test_file_sims}"
            )
            continue

        if train_sims is None or test_sims is None:
            raise AssertionError(f"Missing joint-file split for {name}.")
        required_sims = max(train_sims + test_sims) + 1
        if train_file_sims < required_sims:
            raise ValueError(
                f"{train_path} has {train_file_sims} simulations, but profile "
                f"{config['dataset_profile']!r} requires {required_sims}."
            )
        if set(train_sims) & set(test_sims):
            raise ValueError(f"Train/test simulation overlap detected for {name}.")
        rows.append(
            f"{name}: file={train_file_sims} train_source={len(train_sims)} "
            f"test={len(test_sims)}"
        )

    print("[data] validated files and disjoint simulation splits")
    for row in rows:
        print(f"[data] {row}")


def build_training_module(config: dict) -> SingleStepSupervised:
    model = PDETransformer(
        sample_size=config["sample_size"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        type=config["model_type"],
        patch_size=config["patch_size"],
        periodic=config["periodic"],
        carrier_token_active=config["carrier_token_active"],
        token_mixer_type=config["token_mixer_type"],
        vittt_inner_lr=config["vittt_inner_lr"],
        vittt_head_dim=config["vittt_head_dim"],
        window_ttt_update_mode=config["window_ttt_update_mode"],
        window_ttt_chunk_size=config["window_ttt_chunk_size"],
        ttt_layer_type=config["ttt_layer_type"],
        ttt_mini_batch_size=config["ttt_mini_batch_size"],
        ttt_base_lr=config["ttt_base_lr"],
        ttt_use_gate=config["ttt_use_gate"],
        ttt_scan_checkpoint_group_size=config["ttt_scan_checkpoint_group_size"],
        vittt_padding_mode=config["vittt_padding_mode"],
    )
    module_type = (
        DualEMAValidationSupervised
        if config["use_ema"] and config["ema_dual_validation"]
        else SingleStepSupervised
    )
    monitor = (
        "val/ema_loss_epoch"
        if config["use_ema"] and config["ema_dual_validation"]
        else "val/loss_epoch"
    )
    training_module = module_type(
        model=model,
        image_key=0,
        optimizer="adamw",
        monitor=monitor,
    )
    training_module.learning_rate = config["learning_rate"]
    return training_module


def find_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(
        checkpoint_dir.glob("last*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_initial_weights(module: SingleStepSupervised, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state_dict, strict=True)


def main():
    args = parse_args()
    config = load_config(args.config.expanduser().resolve())
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.devices is not None:
        config["devices"] = args.devices
    if args.strategy is not None:
        config["strategy"] = args.strategy
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.accumulate_grad_batches is not None:
        config["accumulate_grad_batches"] = args.accumulate_grad_batches
    if args.seed is not None:
        config["seed"] = args.seed
    if args.run_root is not None:
        config["run_root"] = str(args.run_root)
    if args.run_name is not None:
        config["run_name"] = args.run_name

    L.seed_everything(config["seed"], workers=True)
    data_dir = Path(config["data_dir"]).expanduser().resolve()
    run_root = Path(config["run_root"]).expanduser().resolve()
    run_dir = run_root / config["run_name"]
    checkpoint_dir = run_dir / "checkpoints"
    module = build_training_module(config)
    parameter_count = sum(parameter.numel() for parameter in module.model.parameters())
    if args.check_config:
        print(OmegaConf.to_yaml(OmegaConf.create(config), resolve=True))
        print(f"parameters: {parameter_count:,}")
        return
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    validate_dataset_files(config)
    if args.check_data:
        return
    data_module = build_data_module(config)
    dual_validation = bool(config["use_ema"] and config["ema_dual_validation"])
    raw_checkpoint = None
    ema_checkpoint = None
    checkpoint = None
    if dual_validation:
        raw_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="raw-epoch-{epoch:03d}",
            monitor="val/raw_loss_epoch",
            mode="min",
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
        )
        ema_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="ema-source-epoch-{epoch:03d}",
            monitor="val/ema_loss_epoch",
            mode="min",
            save_last=False,
            save_top_k=1,
            every_n_epochs=1,
        )
    else:
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch-{epoch:03d}",
            monitor="val/loss_epoch",
            mode="min",
            save_last=True,
            save_top_k=3,
            every_n_epochs=1,
        )
    ema_callback = None
    callbacks = []
    if config["use_ema"]:
        ema_callback = TrainingEMA(
            decay=config["ema_decay"],
            update_every_n_steps=config["ema_update_every_n_steps"],
            validate_with_ema=config["ema_validate"] and not dual_validation,
        )
        callbacks.append(ema_callback)
        if dual_validation:
            if not isinstance(module, DualEMAValidationSupervised):
                raise AssertionError("Dual EMA validation module was not constructed.")
            module.attach_ema_callback(ema_callback)
    if dual_validation:
        callbacks.extend([raw_checkpoint, ema_checkpoint, EpochSummary()])
    else:
        callbacks.extend([checkpoint, EpochSummary()])
    trainer_kwargs = {}
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config["devices"] if torch.cuda.is_available() else 1,
        strategy=config["strategy"],
        precision=config["precision"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(run_root), name=config["run_name"]),
        enable_progress_bar=False,
        log_every_n_steps=10,
        **trainer_kwargs,
    )

    resume_checkpoint = None
    if config["auto_resume"] and not args.fresh:
        resume_checkpoint = find_resume_checkpoint(checkpoint_dir)
    init_checkpoint = None
    if resume_checkpoint is None and config.get("init_checkpoint"):
        init_checkpoint = Path(config["init_checkpoint"]).expanduser().resolve()
        if not init_checkpoint.is_file():
            raise FileNotFoundError(f"Initialization checkpoint does not exist: {init_checkpoint}")
        load_initial_weights(module, init_checkpoint)
    print(OmegaConf.to_yaml(OmegaConf.create(config), resolve=True))
    print(f"parameters: {parameter_count:,}")
    print(
        "effective_global_batch: "
        f"{config['batch_size'] * config['devices'] * config['accumulate_grad_batches']}"
    )
    print(f"resume_checkpoint: {resume_checkpoint}")
    print(f"init_checkpoint: {init_checkpoint}")
    print(
        "validation_weights: "
        f"{'raw+ema' if dual_validation else 'ema' if config['use_ema'] and config['ema_validate'] else 'raw'}"
    )

    trainer.fit(
        module,
        datamodule=data_module,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint else None,
    )
    best_ema_checkpoint = None
    last_ema_checkpoint = None
    raw_best_checkpoint = None
    raw_last_checkpoint = None
    validation = None
    if dual_validation:
        if trainer.is_global_zero:
            raw_best_checkpoint = checkpoint_dir / "raw-best.ckpt"
            shutil.copy2(raw_checkpoint.best_model_path, raw_best_checkpoint)
            raw_last_checkpoint = Path(raw_checkpoint.last_model_path)
            best_ema_checkpoint = export_ema_checkpoint(
                Path(ema_checkpoint.best_model_path),
                checkpoint_dir / "ema-best.ckpt",
            )
            last_ema_checkpoint = export_ema_checkpoint(
                raw_last_checkpoint,
                checkpoint_dir / "ema-last.ckpt",
            )
    else:
        validation = trainer.validate(
            module,
            datamodule=data_module,
            ckpt_path="best",
            verbose=False,
        )
        if ema_callback is not None and trainer.is_global_zero:
            best_ema_checkpoint = export_ema_checkpoint(
                Path(checkpoint.best_model_path),
                checkpoint_dir / "ema-best.ckpt",
            )
            last_ema_checkpoint = export_ema_checkpoint(
                Path(checkpoint.last_model_path),
                checkpoint_dir / "ema-last.ckpt",
            )
    trainer.strategy.barrier()
    if dual_validation:
        print(f"raw_last_checkpoint: {raw_last_checkpoint}")
        print(f"raw_best_checkpoint: {raw_best_checkpoint}")
        print(f"raw_best_source: {raw_checkpoint.best_model_path}")
        print(f"raw_best_validation: {raw_checkpoint.best_model_score}")
        print(f"ema_best_source: {ema_checkpoint.best_model_path}")
        print(f"ema_best_validation: {ema_checkpoint.best_model_score}")
    else:
        print(f"last_checkpoint: {checkpoint.last_model_path}")
        print(f"best_checkpoint: {checkpoint.best_model_path}")
    print(f"ema_last_checkpoint: {last_ema_checkpoint}")
    print(f"ema_best_checkpoint: {best_ema_checkpoint}")
    if validation is not None:
        print(f"best_validation: {validation}")


if __name__ == "__main__":
    main()
