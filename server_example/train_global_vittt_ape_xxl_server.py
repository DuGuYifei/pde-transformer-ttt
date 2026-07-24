"""Focused server entrypoint for fair Attention/ViTTT-PDE comparisons."""

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

# Prefer the reviewed source tree over an older installed wheel on the server.
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

from pdetransformer.core.mixed_channels import PDETransformer, SingleStepSupervised
from pdetransformer.data.pbdl_datatypes.ape_2d_splits import (
    DATASET_PROFILES,
    SEPARATE_TEST_DATASETS,
    ape_2d_xxl_simulation_split,
    dataset_names_for_profile,
)


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
OPTIONAL_CONFIG_KEYS = {"init_checkpoint"}


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
        loss = trainer.callback_metrics.get("val/loss_epoch")
        if loss is not None:
            print(f"[val] epoch={trainer.current_epoch + 1} loss={float(loss):.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--strategy", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulate-grad-batches", type=int)
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing last checkpoint.")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Build the model and print its parameter count without loading data.",
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
    if config["token_mixer_type"] not in {
        "attention",
        "global_vittt",
        "global_h_vittt",
        "global_linear_ttt",
    }:
        raise ValueError(f"Unsupported token_mixer_type={config['token_mixer_type']!r}")
    if config["carrier_token_active"] and config["token_mixer_type"] != "attention":
        raise ValueError("Global ViTTT configurations cannot enable carrier tokens.")
    if config["dataset_profile"] not in DATASET_PROFILES:
        raise ValueError(
            f"Unsupported dataset_profile={config['dataset_profile']!r}; "
            f"expected one of {sorted(DATASET_PROFILES)}."
        )
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
    )
    training_module = SingleStepSupervised(
        model=model,
        image_key=0,
        optimizer="adamw",
        monitor="val/loss_epoch",
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
    data_module = build_data_module(config)
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:03d}",
        monitor="val/loss_epoch",
        mode="min",
        save_last=True,
        save_top_k=3,
        every_n_epochs=1,
    )
    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config["devices"] if torch.cuda.is_available() else 1,
        strategy=config["strategy"],
        precision=config["precision"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[checkpoint, EpochSummary()],
        logger=CSVLogger(save_dir=str(run_root), name=config["run_name"]),
        enable_progress_bar=False,
        log_every_n_steps=10,
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
