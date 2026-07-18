"""Fine-tune Global Linear TTT on a stateless 29-step PDE rollout."""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
for package_name, package_path in {
    "pdetransformer": REPO_ROOT / "pdetransformer",
    "pdetransformer.core": REPO_ROOT / "pdetransformer" / "core",
}.items():
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package

from pdetransformer.core.mixed_channels import (  # noqa: E402
    AutoregressiveRolloutSupervised,
    PDETransformer,
    PersistentAutoregressiveRolloutSupervised,
)


DATASET_NAMES = [
    "diff", "hyp", "burgers", "kdv", "ks", "fisher", "gs_alpha",
    "gs_beta", "gs_gamma", "gs_delta", "gs_epsilon", "gs_theta",
    "gs_iota", "gs_kappa", "sh", "decay_turb", "kolm_flow",
]

REQUIRED_CONFIG_KEYS = {
    "data_dir", "run_root", "run_name", "model_type", "in_channels",
    "out_channels", "patch_size", "periodic", "carrier_token_active",
    "token_mixer_type", "vittt_inner_lr", "vittt_head_dim",
    "vittt_persistent_state",
    "init_checkpoint", "learning_rate", "max_epochs", "batch_size",
    "num_workers", "devices", "strategy", "precision",
    "accumulate_grad_batches", "seed", "downsample_factor", "sample_size",
    "train_unrolling_steps", "train_step_size", "tbptt_chunk_size",
    "test_unrolling_steps", "max_channels", "auto_resume",
}


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
        loss = trainer.callback_metrics.get("val/loss")
        if loss is not None:
            print(f"[val] epoch={trainer.current_epoch + 1} loss={float(loss):.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--strategy", type=str)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    unknown = sorted(set(config) - REQUIRED_CONFIG_KEYS)
    if missing or unknown:
        raise ValueError(f"Invalid config: missing={missing}, unknown={unknown}")
    if config["token_mixer_type"] != "global_linear_ttt":
        raise ValueError("Rollout training requires token_mixer_type=global_linear_ttt")
    if config["carrier_token_active"]:
        raise ValueError("Global Linear TTT cannot enable carrier tokens")
    if config["train_unrolling_steps"] < 2:
        raise ValueError("Rollout training requires at least two steps")
    if config["tbptt_chunk_size"] < 1:
        raise ValueError("tbptt_chunk_size must be positive")
    if not config["init_checkpoint"]:
        raise ValueError("A trained G-L init_checkpoint is required")
    return config


def build_data_module(config: dict):
    from pdetransformer.data import MultiDataModule

    return MultiDataModule(
        path_index={"2D_APE_xxl": str(Path(config["data_dir"]).expanduser())},
        dataset_names=DATASET_NAMES,
        dataset_type="2D_APE_xxl",
        unrolling_steps=config["train_unrolling_steps"],
        train_step_size=config["train_step_size"],
        test_unrolling_steps=config["test_unrolling_steps"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        cache_strategy="none",
        different_resolution_strategy="none",
        normalize_data="mean-std",
        normalize_const="mean-std",
        downsample_factor=config["downsample_factor"],
        max_channels=config["max_channels"],
    )


def build_training_module(config: dict) -> AutoregressiveRolloutSupervised:
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
        vittt_persistent_state=config["vittt_persistent_state"],
    )
    strategy_class = (
        PersistentAutoregressiveRolloutSupervised
        if config["vittt_persistent_state"]
        else AutoregressiveRolloutSupervised
    )
    module = strategy_class(
        model=model,
        image_key=0,
        optimizer="adamw",
        monitor="val/loss",
        train_unrolling_steps=config["train_unrolling_steps"],
        tbptt_chunk_size=config["tbptt_chunk_size"],
        gradient_accumulation_batches=config["accumulate_grad_batches"],
    )
    module.learning_rate = config["learning_rate"]
    return module


def load_initial_weights(module: AutoregressiveRolloutSupervised, path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"G-L initialization checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state_dict, strict=True)
    print(f"initialized G-L from {path}")


def find_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(
        checkpoint_dir.glob("last*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    args = parse_args()
    config = load_config(args.config.expanduser().resolve())
    for key in ("run_name", "max_epochs", "devices", "strategy"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    L.seed_everything(config["seed"], workers=True)
    run_root = Path(config["run_root"]).expanduser().resolve()
    run_dir = run_root / config["run_name"]
    checkpoint_dir = run_dir / "checkpoints"
    resume_checkpoint = None
    if config["auto_resume"] and not args.fresh:
        resume_checkpoint = find_resume_checkpoint(checkpoint_dir)

    module = build_training_module(config)
    total_parameters = sum(parameter.numel() for parameter in module.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    if total_parameters != trainable_parameters:
        raise RuntimeError(
            f"G-L-R29 expects full fine-tuning: total={total_parameters}, "
            f"trainable={trainable_parameters}"
        )

    print(OmegaConf.to_yaml(OmegaConf.create(config), resolve=True))
    print(f"parameters: {total_parameters:,} (all trainable)")
    state_mode = "persistent for full rollout" if config["vittt_persistent_state"] else "reset every PDE step"
    print(f"spatial fast-weight state: {state_mode}")
    print(f"rollout: {config['train_unrolling_steps']} steps")
    print(f"TBPTT detach interval: {config['tbptt_chunk_size']} steps")
    print(f"resume_checkpoint: {resume_checkpoint}")
    if args.check_config:
        return

    if resume_checkpoint is None:
        load_initial_weights(module, Path(config["init_checkpoint"]))

    data_dir = Path(config["data_dir"]).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    data_module = build_data_module(config)
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:03d}",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=3,
        every_n_epochs=1,
    )
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
        accumulate_grad_batches=1,
        callbacks=[checkpoint, EpochSummary()],
        logger=CSVLogger(save_dir=str(run_root), name=config["run_name"]),
        enable_progress_bar=False,
        log_every_n_steps=10,
        **trainer_kwargs,
    )
    trainer.fit(
        module,
        datamodule=data_module,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint else None,
    )
    validation_checkpoint = checkpoint.best_model_path or checkpoint.last_model_path
    validation = trainer.validate(
        module,
        datamodule=data_module,
        ckpt_path=validation_checkpoint,
        verbose=False,
    )
    print(f"last_checkpoint: {checkpoint.last_model_path}")
    print(f"best_checkpoint: {checkpoint.best_model_path}")
    print(f"best_validation: {validation}")


if __name__ == "__main__":
    main()
