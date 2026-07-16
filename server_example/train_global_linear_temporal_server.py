"""Train a frozen Global Linear TTT backbone with persistent latent temporal TTT."""

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
    PDETransformer,
    TemporalRolloutSupervised,
)


DATASET_NAMES = [
    "diff",
    "hyp",
    "burgers",
    "kdv",
    "ks",
    "fisher",
    "gs_alpha",
    "gs_beta",
    "gs_gamma",
    "gs_delta",
    "gs_epsilon",
    "gs_theta",
    "gs_iota",
    "gs_kappa",
    "sh",
    "decay_turb",
    "kolm_flow",
]

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
    "temporal_ttt_enabled",
    "temporal_ttt_layer_type",
    "temporal_ttt_mini_batch_size",
    "temporal_ttt_base_lr",
    "temporal_ttt_gate_init",
    "temporal_ttt_use_output_gate",
    "temporal_ttt_scan_checkpoint_group_size",
    "temporal_ttt_learning_rate",
    "freeze_backbone",
    "init_checkpoint",
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
    "train_step_size",
    "tbptt_chunk_size",
    "test_unrolling_steps",
    "max_channels",
    "auto_resume",
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
    parser.add_argument(
        "--run-name",
        type=str,
        help="Override the configured run name, for example for an isolated smoke run.",
    )
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--strategy", type=str)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--fresh", action="store_true", help="Ignore existing last checkpoints.")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Build the model and validate trainable parameters without loading data.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    unknown = sorted(set(config) - REQUIRED_CONFIG_KEYS)
    if missing or unknown:
        raise ValueError(f"Invalid config: missing={missing}, unknown={unknown}")
    if config["token_mixer_type"] != "global_linear_ttt":
        raise ValueError("G-L-TF training requires token_mixer_type=global_linear_ttt")
    if not config["temporal_ttt_enabled"]:
        raise ValueError("G-L-TF training requires temporal_ttt_enabled=true")
    if not config["freeze_backbone"]:
        raise ValueError("G-L-TF training requires freeze_backbone=true")
    if config["carrier_token_active"]:
        raise ValueError("Global Linear TTT cannot enable carrier tokens")
    if config["train_unrolling_steps"] < 2:
        raise ValueError("Temporal rollout training requires at least two steps")
    if config["train_step_size"] < 1:
        raise ValueError("train_step_size must be positive")
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


def build_training_module(config: dict) -> TemporalRolloutSupervised:
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
        temporal_ttt_enabled=config["temporal_ttt_enabled"],
        temporal_ttt_layer_type=config["temporal_ttt_layer_type"],
        temporal_ttt_mini_batch_size=config["temporal_ttt_mini_batch_size"],
        temporal_ttt_base_lr=config["temporal_ttt_base_lr"],
        temporal_ttt_gate_init=config["temporal_ttt_gate_init"],
        temporal_ttt_use_output_gate=config["temporal_ttt_use_output_gate"],
        temporal_ttt_scan_checkpoint_group_size=config[
            "temporal_ttt_scan_checkpoint_group_size"
        ],
    )
    module = TemporalRolloutSupervised(
        model=model,
        image_key=0,
        optimizer="adamw",
        monitor="val/loss",
        train_unrolling_steps=config["train_unrolling_steps"],
        tbptt_chunk_size=config["tbptt_chunk_size"],
        gradient_accumulation_batches=config["accumulate_grad_batches"],
        temporal_ttt_learning_rate=config["temporal_ttt_learning_rate"],
    )
    module.learning_rate = config["learning_rate"]
    return module


def initialize_from_gl_checkpoint(
    module: TemporalRolloutSupervised,
    checkpoint_path: Path,
) -> None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"G-L initialization checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload in {checkpoint_path}")
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    invalid_missing = [key for key in missing if ".temporal_ttt." not in key]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the G-L backbone: "
            f"invalid_missing={invalid_missing[:8]} unexpected={unexpected[:8]}"
        )
    if not missing:
        raise RuntimeError("Expected new temporal_ttt parameters to be absent from G-L checkpoint")
    print(f"initialized G-L backbone from {checkpoint_path}")
    print(f"new temporal parameter tensors: {len(missing)}")


def freeze_gl_backbone(module: TemporalRolloutSupervised) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    temporal = module.model.model.temporal_ttt
    if temporal is None:
        raise RuntimeError("Temporal module was not constructed")
    for parameter in temporal.parameters():
        parameter.requires_grad_(True)

    invalid_trainable = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and ".temporal_ttt." not in name
    ]
    if invalid_trainable:
        raise RuntimeError(f"Backbone freeze failed: {invalid_trainable[:8]}")


def parameter_summary(module: TemporalRolloutSupervised) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    if trainable == 0 or trainable == total:
        raise RuntimeError(f"Unexpected frozen parameter split: trainable={trainable} total={total}")
    return total, trainable


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
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.devices is not None:
        config["devices"] = args.devices
    if args.strategy is not None:
        config["strategy"] = args.strategy

    L.seed_everything(config["seed"], workers=True)
    run_root = Path(config["run_root"]).expanduser().resolve()
    run_dir = run_root / config["run_name"]
    checkpoint_dir = run_dir / "checkpoints"
    resume_checkpoint = None
    if config["auto_resume"] and not args.fresh:
        resume_checkpoint = find_resume_checkpoint(checkpoint_dir)

    module = build_training_module(config)
    init_checkpoint = Path(config["init_checkpoint"]).expanduser()
    if resume_checkpoint is None:
        initialize_from_gl_checkpoint(module, init_checkpoint)
    else:
        print(f"skipping G-L initialization because resume checkpoint exists: {resume_checkpoint}")
    freeze_gl_backbone(module)
    total_parameters, trainable_parameters = parameter_summary(module)

    print(OmegaConf.to_yaml(OmegaConf.create(config), resolve=True))
    print(f"total parameters: {total_parameters:,}")
    print(f"trainable temporal parameters: {trainable_parameters:,}")
    print("temporal state lifetime: full rollout")
    print(f"TBPTT detach interval: {config['tbptt_chunk_size']} steps")
    if args.check_config:
        return

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

    print(f"resume_checkpoint: {resume_checkpoint}")

    trainer.fit(
        module,
        datamodule=data_module,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint else None,
    )
    validation = trainer.validate(
        module,
        datamodule=data_module,
        ckpt_path="best",
        verbose=False,
    )
    print(f"last_checkpoint: {checkpoint.last_model_path}")
    print(f"best_checkpoint: {checkpoint.best_model_path}")
    print(f"best_validation: {validation}")


if __name__ == "__main__":
    main()
