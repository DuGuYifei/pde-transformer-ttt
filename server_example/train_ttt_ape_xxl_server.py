"""Server training entrypoint matching ipynb_example/train_ttt_ape_xxl.ipynb.

Expected copied layout from Kaggle:
    ~/working/datasets/*.hdf5
    ~/working/ttt_cache_experiments/train_once/checkpoints/last.ckpt

Recommended first run on GTX 1080 Ti:
    CUDA_VISIBLE_DEVICES=0 python train_ttt_ape_xxl_server.py --config train_ttt_ape_xxl_server.yaml

Optional 2-GPU run after smoke_2gpu.py succeeds:
    CUDA_VISIBLE_DEVICES=0,1 python train_ttt_ape_xxl_server.py --config train_ttt_ape_xxl_server.yaml --devices 2 --strategy ddp --accumulate-grad-batches 4
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from pdetransformer.core.mixed_channels import PDETransformer, SingleStepSupervised
from pdetransformer.data import MultiDataModule


FULL_DATASET_NAMES = [
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


DEFAULT_CONFIG: dict[str, Any] = {
    "work_dir": "~/working",
    "data_dir": None,
    "run_root": None,
    "run_name": "train_once",
    "max_epochs": 100,
    "batch_size": 8,
    "num_workers": 2,
    "devices": 1,
    "strategy": "auto",
    "precision": "32-true",
    "accumulate_grad_batches": 8,
    "seed": 42,
    "resume": False,
    "auto_resume": True,
    "checkpoint_path": None,
    "skip_rollout_eval": False,
    "generate_data": False,
    "low_res": True,
    "generation_num_sims_default": 60,
    "generation_num_sims_gs_small": 10,
    "generation_num_sims_gs_full": 100,
    "generation_num_sims_test": 5,
    "generation_num_sims_gs_test": 3,
    "generation_gpu_id": "0",
    "render_only": False,
    "force_regenerate_data": False,
}


class EpochPrintCallback(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.time()

    def _scalar_metrics(self, trainer, prefixes):
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if not any(key.startswith(prefix) or prefix in key for prefix in prefixes):
                continue
            if hasattr(value, "detach") and value.numel() == 1:
                metrics[key] = float(value.detach().cpu())
        return metrics

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - getattr(self, "_epoch_start_time", time.time())
        metrics = self._scalar_metrics(trainer, prefixes=("train", "loss"))
        metric_text = ", ".join(f"{k}={v:.6g}" for k, v in sorted(metrics.items())) or "no train metrics yet"
        print(
            f"[train] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"global_step={trainer.global_step} elapsed={elapsed / 60:.1f} min | {metric_text}"
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = self._scalar_metrics(trainer, prefixes=("val",))
        metric_text = ", ".join(f"{k}={v:.6g}" for k, v in sorted(metrics.items())) or "no val metrics yet"
        print(f"[val] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} global_step={trainer.global_step} | {metric_text}")


def _path_or_none(value: str | None) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    return Path(value)


def _load_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if loaded is None:
        return config
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    unknown = sorted(set(loaded) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {unknown}")
    config.update(loaded)
    return config


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, remaining = config_parser.parse_known_args()
    config = _load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Train PDE Transformer TTT on server.",
        parents=[config_parser],
    )
    parser.set_defaults(config=config_args.config)
    parser.add_argument("--work-dir", type=Path, default=_path_or_none(config["work_dir"]))
    parser.add_argument("--data-dir", type=Path, default=_path_or_none(config["data_dir"]))
    parser.add_argument("--run-root", type=Path, default=_path_or_none(config["run_root"]))
    parser.add_argument("--run-name", type=str, default=config["run_name"])
    parser.add_argument("--max-epochs", type=int, default=config["max_epochs"])
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--num-workers", type=int, default=config["num_workers"])
    parser.add_argument("--devices", type=int, default=config["devices"])
    parser.add_argument("--strategy", type=str, default=config["strategy"])
    parser.add_argument("--precision", type=str, default=config["precision"])
    parser.add_argument("--accumulate-grad-batches", type=int, default=config["accumulate_grad_batches"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=config["resume"], help="Resume from --checkpoint-path.")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=config["auto_resume"], help="Resume if --checkpoint-path exists.")
    parser.add_argument("--checkpoint-path", type=Path, default=_path_or_none(config["checkpoint_path"]))
    parser.add_argument("--skip-rollout-eval", action=argparse.BooleanOptionalAction, default=config["skip_rollout_eval"])
    parser.add_argument("--generate-data", action=argparse.BooleanOptionalAction, default=config["generate_data"])
    parser.add_argument("--low-res", action=argparse.BooleanOptionalAction, default=config["low_res"])
    parser.add_argument("--generation-num-sims-default", type=int, default=config["generation_num_sims_default"])
    parser.add_argument("--generation-num-sims-gs-small", type=int, default=config["generation_num_sims_gs_small"])
    parser.add_argument("--generation-num-sims-gs-full", type=int, default=config["generation_num_sims_gs_full"])
    parser.add_argument("--generation-num-sims-test", type=int, default=config["generation_num_sims_test"])
    parser.add_argument("--generation-num-sims-gs-test", type=int, default=config["generation_num_sims_gs_test"])
    parser.add_argument("--generation-gpu-id", type=str, default=config["generation_gpu_id"])
    parser.add_argument("--render-only", action=argparse.BooleanOptionalAction, default=config["render_only"])
    parser.add_argument("--force-regenerate-data", action=argparse.BooleanOptionalAction, default=config["force_regenerate_data"])
    return parser.parse_args(remaining)


def run_generation(
    args: argparse.Namespace,
    data_dir: Path,
    pde: str,
    out_name: str | None = None,
    num_sims: int = 60,
    test_set: bool = False,
) -> None:
    out_name = out_name or pde
    target_hdf5 = data_dir / f"{out_name}.hdf5"
    if target_hdf5.exists() and not args.force_regenerate_data:
        print(f"[data] skip existing {target_hdf5}")
        return

    cmd = [
        sys.executable,
        "-m",
        "pdetransformer.data.simulations_apebench.simulation",
        "--pde",
        pde,
        "--out_name",
        out_name,
        "--out_path",
        str(data_dir),
        "--num_sims",
        str(num_sims),
        "--gpu_id",
        str(args.generation_gpu_id),
    ]
    if test_set:
        cmd.append("--test_set")
    if args.low_res:
        cmd.append("--low-res")
    if args.render_only:
        cmd.append("--render_only")

    print("[data] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def generate_ape_xxl_data(args: argparse.Namespace, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    print("[data] generation enabled")
    print("[data] output:", data_dir)
    print("[data] low_res:", args.low_res)
    print("[data] force_regenerate_data:", args.force_regenerate_data)

    for pde in ["diff", "hyp", "burgers", "kdv", "fisher", "sh"]:
        run_generation(args, data_dir, pde, num_sims=args.generation_num_sims_default)

    for pde in ["ks", "decay_turb", "kolm_flow"]:
        run_generation(args, data_dir, pde, num_sims=args.generation_num_sims_default)
        run_generation(args, data_dir, pde, out_name=f"{pde}_test", num_sims=args.generation_num_sims_test, test_set=True)

    for pde in ["gs_alpha", "gs_beta", "gs_gamma", "gs_epsilon"]:
        run_generation(args, data_dir, pde, num_sims=args.generation_num_sims_gs_small)
        run_generation(args, data_dir, pde, out_name=f"{pde}_test", num_sims=args.generation_num_sims_gs_test, test_set=True)

    for pde in ["gs_delta", "gs_theta", "gs_iota", "gs_kappa"]:
        run_generation(args, data_dir, pde, num_sims=args.generation_num_sims_gs_full)


def build_data_module(data_dir: Path, dataset_names: list[str], batch_size: int, num_workers: int) -> MultiDataModule:
    params_data = {
        "path_index": {"2D_APE_xxl": str(data_dir)},
        "dataset_names": dataset_names,
        "dataset_type": "2D_APE_xxl",
        "unrolling_steps": 1,
        "test_unrolling_steps": 29,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "cache_strategy": "none",
        "different_resolution_strategy": "none",
        "normalize_data": "mean-std",
        "normalize_const": "mean-std",
        "downsample_factor": 2,
        "max_channels": 2,
    }
    return MultiDataModule(**params_data)


def build_strategy(seed: int, use_ttt_state_cache_inference: bool = False) -> SingleStepSupervised:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = PDETransformer(
        sample_size=128,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        patch_size=4,
        periodic=True,
        carrier_token_active=False,
        use_ttt_window_attention=True,
        ttt_layer_type="linear",
        ttt_mini_batch_size=16,
        ttt_base_lr=1.0,
    )
    strategy = SingleStepSupervised(
        model=model,
        image_key=0,
        optimizer="adamw",
        use_ttt_state_cache_inference=use_ttt_state_cache_inference,
        use_ttt_state_cache_train=False,
    )
    strategy.learning_rate = 4.0e-5
    return strategy


def build_trainer(args: argparse.Namespace, run_root: Path) -> L.Trainer:
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_root / args.run_name / "checkpoints",
        filename="epoch-{epoch:03d}",
        monitor="val/loss_epoch",
        mode="min",
        save_last=True,
        save_top_k=3,
        every_n_epochs=1,
    )
    return L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        strategy=args.strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback, EpochPrintCallback()],
        logger=CSVLogger(save_dir=str(run_root), name=args.run_name),
        enable_progress_bar=False,
        log_every_n_steps=10,
    )


def run_rollout_check(strategy, datamodule, use_ttt_state_cache_inference: bool, num_frames: int = 29) -> dict:
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    strategy.use_ttt_state_cache_inference = use_ttt_state_cache_inference
    strategy.use_ttt_state_cache_train = False
    strategy = strategy.to(device)
    strategy.eval()
    with torch.no_grad():
        prediction, reference = strategy.predict(batch, device=device, num_frames=num_frames)

    prediction_np = np.asarray(prediction)
    reference_np = np.asarray(reference)
    rollout_mse = float(np.mean((prediction_np - reference_np) ** 2))
    return {
        "use_ttt_state_cache_inference": use_ttt_state_cache_inference,
        "use_ttt_state_cache_train": False,
        "prediction_shape": tuple(prediction_np.shape),
        "reference_shape": tuple(reference_np.shape),
        "rollout_mse": rollout_mse,
    }


def find_latest_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoints = sorted(
        checkpoint_dir.glob("last*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if last_checkpoints:
        return last_checkpoints[0]
    return checkpoint_dir / "last.ckpt"


def print_checkpoint_listing(run_root: Path) -> None:
    candidates = [
        run_root / "train_once" / "checkpoints",
        run_root / "inference_cache_off" / "checkpoints",
        run_root / "inference_cache_on" / "checkpoints",
        Path("./lightning_logs/version_0/checkpoints"),
        Path("./lightning_logs/version_1/checkpoints"),
    ]
    for ckpt_dir in candidates:
        print("\nDIR:", ckpt_dir.resolve())
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*.ckpt"))
            if ckpts:
                for path in ckpts:
                    print(f"  {path.name} | {path.stat().st_size / 1024**2:.1f} MB")
            else:
                print("  exists, but no .ckpt files")
        else:
            print("  not found")


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.expanduser().resolve()
    data_dir = (args.data_dir or work_dir / "datasets").expanduser().resolve()
    run_root = (args.run_root or work_dir / "ttt_cache_experiments").expanduser().resolve()
    checkpoint_dir = run_root / args.run_name / "checkpoints"
    checkpoint_path = (
        args.checkpoint_path.expanduser().resolve()
        if args.checkpoint_path
        else find_latest_last_checkpoint(checkpoint_dir).expanduser().resolve()
    )

    print("config:", args.config)
    print("work_dir:", work_dir)
    print("data_dir:", data_dir)
    print("run_root:", run_root)
    print("checkpoint_path:", checkpoint_path)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
    print("precision:", args.precision)
    print("devices:", args.devices, "strategy:", args.strategy)
    print("accumulate_grad_batches:", args.accumulate_grad_batches)
    print("generate_data:", args.generate_data, "low_res:", args.low_res)

    run_root.mkdir(parents=True, exist_ok=True)
    if args.generate_data:
        generate_ape_xxl_data(args, data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    data_module = build_data_module(data_dir, FULL_DATASET_NAMES, args.batch_size, args.num_workers)
    data_module.setup(stage="fit")

    strategy = build_strategy(args.seed, use_ttt_state_cache_inference=False)
    trainer = build_trainer(args, run_root)

    should_resume = args.resume or (args.auto_resume and checkpoint_path.exists())
    ckpt_path = str(checkpoint_path) if should_resume else None
    print("resuming from checkpoint:" if ckpt_path else "starting fresh training run", ckpt_path or "")

    trainer.fit(strategy, datamodule=data_module, ckpt_path=ckpt_path)
    val_metrics = trainer.validate(strategy, datamodule=data_module, verbose=False)

    checkpoint_callback = trainer.checkpoint_callback
    print("last checkpoint:", checkpoint_callback.last_model_path)
    print("best checkpoints:", checkpoint_callback.best_k_models)
    print("validation:", val_metrics)
    print_checkpoint_listing(run_root)

    if args.skip_rollout_eval:
        return

    experiment_results = []
    for name, use_cache in [("inference_cache_off", False), ("inference_cache_on", True)]:
        print(f"\n=== Evaluating {name} ===")
        result = {
            "name": name,
            "checkpoint": trainer.checkpoint_callback.last_model_path,
            "max_epochs": args.max_epochs,
            "val_metrics": val_metrics,
            **run_rollout_check(strategy, data_module, use_cache, num_frames=29),
        }
        experiment_results.append(result)
        print(json.dumps(result, indent=2))

    off = next(r for r in experiment_results if not r["use_ttt_state_cache_inference"])
    on = next(r for r in experiment_results if r["use_ttt_state_cache_inference"])
    print("cache off rollout mse:", off["rollout_mse"])
    print("cache on  rollout mse:", on["rollout_mse"])
    print("prediction shapes:", off["prediction_shape"], on["prediction_shape"])

    del strategy, data_module
    gc.collect()


if __name__ == "__main__":
    main()
