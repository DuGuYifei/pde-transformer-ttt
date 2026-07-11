"""Server training entrypoint matching ipynb_example/train_ttt_ape_xxl.ipynb.

Expected copied layout from Kaggle:
    ~/working/datasets/*.hdf5
    ~/working/ttt_cache_experiments/train_cache_on/checkpoints/last.ckpt

Recommended first run on GTX 1080 Ti:
    CUDA_VISIBLE_DEVICES=0 python server_example/train_ttt_ape_xxl_server.py --config server_example/pdes_vittt-cacheoff_128_60sims.yaml

Optional 2-GPU run after smoke_2gpu.py succeeds:
    CUDA_VISIBLE_DEVICES=0,1 python server_example/train_ttt_ape_xxl_server.py --config server_example/pdes_vittt-cacheoff_128_60sims.yaml --devices 2 --strategy ddp --accumulate-grad-batches 8
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

# Server folders may also have an older wheel installed in the active venv.
# Always resolve project imports from the source tree containing this script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pdetransformer.core.mixed_channels import (
    PDETransformer,
    SingleStepSupervised,
    TemporalRolloutSupervised,
)
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
    "model_type": "PDE-S",
    "in_channels": 2,
    "out_channels": 2,
    "patch_size": 4,
    "periodic": True,
    "carrier_token_active": False,
    # False builds the plain PDE-S/B/L without TTT window attention (the
    # original architecture; verified computationally identical to upstream).
    "use_ttt_window_attention": True,
    # Preferred explicit mixer selector. null keeps backward compatibility:
    # use_ttt_window_attention=false -> attention, true -> ttt_sequence.
    "token_mixer_type": None,
    "use_ttt_state_cache_train": True,
    "ttt_layer_type": "linear",
    "ttt_mini_batch_size": 16,
    "ttt_base_lr": 1.0,
    "ttt_use_gate": False,
    "ttt_scan_checkpoint_group_size": 0,
    "vittt_inner_lr": 1.0,
    "vittt_padding_mode": "zero",
    "attention_ttt_type": "ttt_sequence",
    "attention_ttt_gate_init": 0.1,
    "attention_ttt_bidirectional": True,
    "global_ttt_stage_names": [],
    "global_ttt_inner_lr": 1.0,
    "global_ttt_gate_init": 0.0,
    "global_ttt_key_norm": True,
    "training_mode": "single_step",
    "train_unrolling_steps": 1,
    "train_step_size": 1,
    "tbptt_chunk_size": 4,
    "temporal_ttt_enabled": False,
    "temporal_ttt_layer_type": "mlp",
    "temporal_ttt_mini_batch_size": 64,
    "temporal_ttt_base_lr": 1.0,
    "temporal_ttt_gate_init": 0.1,
    "temporal_ttt_learning_rate": 1.0e-4,
    "temporal_ttt_use_output_gate": False,
    "temporal_ttt_scan_checkpoint_group_size": 0,
    "learning_rate": 4.0e-5,
    "max_epochs": 100,
    "batch_size": 8,
    "num_workers": 2,
    # Data resolution: hdf5 files are 256x256; the loader avg-pools by this
    # factor before batching. 2 = the low-res (128) experiment series, 1 = native 256.
    "downsample_factor": 2,
    # Stored in the model config (metadata for diffusers pipelines; supervised
    # training does not consume it). Keep in sync with the effective input size.
    "sample_size": 128,
    "test_unrolling_steps": 29,
    "max_channels": 2,
    # Optional sim-range overrides for the 2D_APE_xxl split, as [start, end).
    # null keeps the package defaults (joint train [0,50] / test = whole file,
    # gs_delta-group train [0,80] / test = whole file). For the 600-sims
    # ape2d-full corpus set e.g. joint [0,500] / [500,600].
    "sims_split_joint_train": None,
    "sims_split_joint_test": None,
    "sims_split_gs_train": None,
    "sims_split_gs_test": None,
    "devices": 1,
    "strategy": "auto",
    "precision": "32-true",
    "accumulate_grad_batches": 8,
    "seed": 42,
    "resume": False,
    "auto_resume": True,
    "checkpoint_path": None,
    "init_checkpoint_path": None,
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


def resolve_token_mixer_type(token_mixer_type: str | None, use_ttt_window_attention: bool) -> str:
    if token_mixer_type is not None:
        return token_mixer_type
    return "ttt_sequence" if use_ttt_window_attention else "attention"


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
    parser.add_argument("--model-type", type=str, choices=("PDE-S", "PDE-B", "PDE-L"), default=config["model_type"])
    parser.add_argument("--in-channels", type=int, default=config["in_channels"])
    parser.add_argument("--out-channels", type=int, default=config["out_channels"])
    parser.add_argument("--patch-size", type=int, default=config["patch_size"])
    parser.add_argument("--periodic", action=argparse.BooleanOptionalAction, default=config["periodic"])
    parser.add_argument("--carrier-token-active", action=argparse.BooleanOptionalAction, default=config["carrier_token_active"])
    parser.add_argument(
        "--use-ttt-window-attention",
        action=argparse.BooleanOptionalAction,
        default=config["use_ttt_window_attention"],
        help="Build the model with TTT window attention. --no-use-ttt-window-attention trains the plain baseline.",
    )
    parser.add_argument(
        "--token-mixer-type",
        type=str,
        choices=("attention", "ttt_sequence", "vittt", "attention_ttt"),
        default=config["token_mixer_type"],
        help="Explicit local token mixer. Omit/null for legacy use_ttt_window_attention mapping.",
    )
    parser.add_argument(
        "--use-ttt-state-cache-train",
        action=argparse.BooleanOptionalAction,
        default=config["use_ttt_state_cache_train"],
        help="Use TTT state cache during training and validation forward passes.",
    )
    parser.add_argument(
        "--ttt-layer-type",
        type=str,
        choices=("linear", "mlp"),
        default=config["ttt_layer_type"],
        help="TTT inner learner type used inside TTT window attention.",
    )
    parser.add_argument("--ttt-mini-batch-size", type=int, default=config["ttt_mini_batch_size"])
    parser.add_argument("--ttt-base-lr", type=float, default=config["ttt_base_lr"])
    parser.add_argument("--ttt-use-gate", action=argparse.BooleanOptionalAction, default=config["ttt_use_gate"])
    parser.add_argument("--ttt-scan-checkpoint-group-size", type=int, default=config["ttt_scan_checkpoint_group_size"])
    parser.add_argument("--vittt-inner-lr", type=float, default=config["vittt_inner_lr"])
    parser.add_argument(
        "--vittt-padding-mode",
        type=str,
        choices=("zero", "replicate"),
        default=config["vittt_padding_mode"],
    )
    parser.add_argument(
        "--attention-ttt-type",
        type=str,
        choices=("ttt_sequence", "vittt"),
        default=config["attention_ttt_type"],
        help="Post-attention TTT branch used when token_mixer_type=attention_ttt.",
    )
    parser.add_argument("--attention-ttt-gate-init", type=float, default=config["attention_ttt_gate_init"])
    parser.add_argument(
        "--attention-ttt-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=config["attention_ttt_bidirectional"],
        help="Apply reverse TTT after forward TTT, matching video-dit style.",
    )
    parser.add_argument("--global-ttt-inner-lr", type=float, default=config["global_ttt_inner_lr"])
    parser.add_argument("--global-ttt-gate-init", type=float, default=config["global_ttt_gate_init"])
    parser.add_argument(
        "--global-ttt-key-norm",
        action=argparse.BooleanOptionalAction,
        default=config["global_ttt_key_norm"],
    )
    parser.add_argument(
        "--training-mode",
        choices=("single_step", "temporal_rollout"),
        default=config["training_mode"],
    )
    parser.add_argument("--train-unrolling-steps", type=int, default=config["train_unrolling_steps"])
    parser.add_argument("--train-step-size", type=int, default=config["train_step_size"])
    parser.add_argument("--tbptt-chunk-size", type=int, default=config["tbptt_chunk_size"])
    parser.add_argument(
        "--temporal-ttt-enabled",
        action=argparse.BooleanOptionalAction,
        default=config["temporal_ttt_enabled"],
    )
    parser.add_argument(
        "--temporal-ttt-layer-type",
        choices=("linear", "mlp"),
        default=config["temporal_ttt_layer_type"],
    )
    parser.add_argument(
        "--temporal-ttt-mini-batch-size",
        type=int,
        default=config["temporal_ttt_mini_batch_size"],
    )
    parser.add_argument("--temporal-ttt-base-lr", type=float, default=config["temporal_ttt_base_lr"])
    parser.add_argument("--temporal-ttt-gate-init", type=float, default=config["temporal_ttt_gate_init"])
    parser.add_argument(
        "--temporal-ttt-learning-rate",
        type=float,
        default=config["temporal_ttt_learning_rate"],
    )
    parser.add_argument(
        "--temporal-ttt-use-output-gate",
        action=argparse.BooleanOptionalAction,
        default=config["temporal_ttt_use_output_gate"],
    )
    parser.add_argument(
        "--temporal-ttt-scan-checkpoint-group-size",
        type=int,
        default=config["temporal_ttt_scan_checkpoint_group_size"],
    )
    parser.add_argument("--learning-rate", type=float, default=config["learning_rate"])
    parser.add_argument("--max-epochs", type=int, default=config["max_epochs"])
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--num-workers", type=int, default=config["num_workers"])
    parser.add_argument("--downsample-factor", type=int, default=config["downsample_factor"])
    parser.add_argument("--sample-size", type=int, default=config["sample_size"])
    parser.add_argument("--test-unrolling-steps", type=int, default=config["test_unrolling_steps"])
    parser.add_argument("--max-channels", type=int, default=config["max_channels"])
    parser.add_argument("--devices", type=int, default=config["devices"])
    parser.add_argument("--strategy", type=str, default=config["strategy"])
    parser.add_argument("--precision", type=str, default=config["precision"])
    parser.add_argument("--accumulate-grad-batches", type=int, default=config["accumulate_grad_batches"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=config["resume"], help="Resume from --checkpoint-path.")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=config["auto_resume"], help="Resume if --checkpoint-path exists.")
    parser.add_argument("--checkpoint-path", type=Path, default=_path_or_none(config["checkpoint_path"]))
    parser.add_argument(
        "--init-checkpoint-path",
        type=Path,
        default=_path_or_none(config["init_checkpoint_path"]),
        help="Initialize matching weights without restoring optimizer or epoch state.",
    )
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
    # yaml-only keys (list-valued; no CLI flags) still need to land on the namespace
    parser.set_defaults(
        sims_split_joint_train=config["sims_split_joint_train"],
        sims_split_joint_test=config["sims_split_joint_test"],
        sims_split_gs_train=config["sims_split_gs_train"],
        sims_split_gs_test=config["sims_split_gs_test"],
        global_ttt_stage_names=config["global_ttt_stage_names"],
    )
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


def build_data_module(
    data_dir: Path,
    dataset_names: list[str],
    batch_size: int,
    num_workers: int,
    downsample_factor: int = 2,
    train_unrolling_steps: int = 1,
    train_step_size: int = 1,
    test_unrolling_steps: int = 29,
    max_channels: int = 2,
) -> MultiDataModule:
    params_data = {
        "path_index": {"2D_APE_xxl": str(data_dir)},
        "dataset_names": dataset_names,
        "dataset_type": "2D_APE_xxl",
        "unrolling_steps": train_unrolling_steps,
        "train_step_size": train_step_size,
        "test_unrolling_steps": test_unrolling_steps,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "cache_strategy": "none",
        "different_resolution_strategy": "none",
        "normalize_data": "mean-std",
        "normalize_const": "mean-std",
        "downsample_factor": downsample_factor,
        "max_channels": max_channels,
    }
    return MultiDataModule(**params_data)


# Datasets the installed package routes through the joint-file branch of
# ape_2d_xxl_datasets (hardcoded sim splits). Everything else (separate *_test
# files) is untouched by the override below.
_JOINT_GS_DATASETS = ("gs_delta", "gs_theta", "gs_iota", "gs_kappa")
_JOINT_DEFAULT_DATASETS = ("adv", "diff", "adv_diff", "disp", "hyp", "burgers", "kdv", "fisher", "sh")


def apply_sims_split_override(
    joint_train: list[int] | None,
    joint_test: list[int] | None,
    gs_train: list[int] | None,
    gs_test: list[int] | None,
) -> None:
    """Make the package's hardcoded 2D_APE_xxl sim splits yaml-configurable.

    The installed pdetransformer wheel hardcodes train sims [0,50) (joint files)
    and [0,80) (gs_delta group), with the test set spanning the whole file.
    Rather than forking the wheel, rebind the dataset factory that
    pbdl_module.get_datasets resolves at call time. Datasets with separate
    *_test files keep the original code path. Ranges are [start, end).
    """
    from pdetransformer.data import pbdl_module
    from pdetransformer.data.pbdl_datatypes import ape_2d_xxl as _orig
    from pdetransformer.data.pbdl_dataloader.dataset import Dataset as PBDLDataset
    from pdetransformer.data.pbdl_datatypes.variable_dt_dataset import VariableDtDataset
    from torch.utils.data import random_split

    split_seed = 46  # identical to the package's train/val split seed

    def _ranged(bounds: list[int] | None, fallback: tuple[int, int]) -> list[int]:
        start, end = bounds if bounds is not None else fallback
        return list(range(start, end))

    def datasets_fn(
        dataset_name: str,
        dataset_directory: str,
        unrolling_steps: int,
        intermediate_time_steps: bool = True,
        variable_dt_stride_maximum: int = 1,
        test_variable_dt_stride_maximum: int = 1,
        test_unrolling_steps: int | None = None,
        test_intermediate_time_steps: bool | None = None,
        normalize_data: str | None = None,
        normalize_const: str | None = None,
        **kwargs,
    ):
        train_step_size = kwargs.pop("train_step_size", None)
        if dataset_name in _JOINT_GS_DATASETS:
            train_sims = _ranged(gs_train, (0, 80))
            test_sims = list(range(*gs_test)) if gs_test is not None else None
        elif dataset_name in _JOINT_DEFAULT_DATASETS:
            train_sims = _ranged(joint_train, (0, 50))
            test_sims = list(range(*joint_test)) if joint_test is not None else None
        else:
            # separate-test-file datasets: original behavior, untouched
            return _orig.ape_2d_xxl_datasets(
                dataset_name, dataset_directory, unrolling_steps,
                train_step_size=train_step_size,
                intermediate_time_steps=intermediate_time_steps,
                variable_dt_stride_maximum=variable_dt_stride_maximum,
                test_variable_dt_stride_maximum=test_variable_dt_stride_maximum,
                test_unrolling_steps=test_unrolling_steps,
                test_intermediate_time_steps=test_intermediate_time_steps,
                normalize_data=normalize_data,
                normalize_const=normalize_const,
                **kwargs,
            )

        if test_unrolling_steps is None:
            test_unrolling_steps = unrolling_steps
        if test_intermediate_time_steps is None:
            test_intermediate_time_steps = intermediate_time_steps
        if test_variable_dt_stride_maximum is None:
            test_variable_dt_stride_maximum = variable_dt_stride_maximum

        norm_const = normalize_const if "gs_" not in dataset_name else None
        params_train = {
            "dset_name": dataset_name,
            "local_datasets_dir": dataset_directory,
            "sel_sims": train_sims,
            "time_steps": unrolling_steps,
            "step_size": train_step_size,
            "intermediate_time_steps": intermediate_time_steps,
            "normalize_const": norm_const,
            "normalize_data": normalize_data,
        }
        if variable_dt_stride_maximum <= 1:
            pbdl_all = PBDLDataset(**params_train)
        else:
            pbdl_all = VariableDtDataset(
                **params_train, maximum_dt=variable_dt_stride_maximum, seed=None
            )

        train, val = random_split(
            pbdl_all, [0.85, 0.15], generator=torch.Generator().manual_seed(split_seed)
        )
        train.indices = sorted(train.indices)
        val.indices = sorted(val.indices)

        params_test = {
            "dset_name": dataset_name,
            "local_datasets_dir": dataset_directory,
            "time_steps": test_unrolling_steps,
            "intermediate_time_steps": test_intermediate_time_steps,
            "normalize_const": norm_const,
            "normalize_data": normalize_data,
        }
        if test_sims is not None:
            params_test["sel_sims"] = test_sims
        if test_variable_dt_stride_maximum <= 1:
            test = PBDLDataset(**params_test)
        else:
            test = VariableDtDataset(
                **params_test, maximum_dt=test_variable_dt_stride_maximum, seed=split_seed
            )

        return train, val, test

    pbdl_module.ape_2d_xxl_datasets = datasets_fn
    print(
        "[data] sims split override active:",
        f"joint_train={joint_train or [0, 50]} joint_test={joint_test or 'whole-file'}",
        f"gs_train={gs_train or [0, 80]} gs_test={gs_test or 'whole-file'}",
    )


def build_strategy(
    seed: int,
    model_type: str = "PDE-S",
    use_ttt_state_cache_inference: bool = False,
    use_ttt_state_cache_train: bool = True,
    sample_size: int = 128,
    in_channels: int = 2,
    out_channels: int = 2,
    patch_size: int = 4,
    periodic: bool = True,
    carrier_token_active: bool = False,
    use_ttt_window_attention: bool = True,
    token_mixer_type: str | None = None,
    ttt_layer_type: str = "linear",
    ttt_mini_batch_size: int = 16,
    ttt_base_lr: float = 1.0,
    ttt_use_gate: bool = False,
    ttt_scan_checkpoint_group_size: int = 0,
    vittt_inner_lr: float = 1.0,
    vittt_padding_mode: str = "zero",
    attention_ttt_type: str = "ttt_sequence",
    attention_ttt_gate_init: float = 0.1,
    attention_ttt_bidirectional: bool = True,
    global_ttt_stage_names: list[str] | None = None,
    global_ttt_inner_lr: float = 1.0,
    global_ttt_gate_init: float = 0.0,
    global_ttt_key_norm: bool = True,
    training_mode: str = "single_step",
    train_unrolling_steps: int = 1,
    tbptt_chunk_size: int = 4,
    gradient_accumulation_batches: int = 1,
    temporal_ttt_enabled: bool = False,
    temporal_ttt_layer_type: str = "mlp",
    temporal_ttt_mini_batch_size: int = 64,
    temporal_ttt_base_lr: float = 1.0,
    temporal_ttt_gate_init: float = 0.1,
    temporal_ttt_learning_rate: float = 1.0e-4,
    temporal_ttt_use_output_gate: bool = False,
    temporal_ttt_scan_checkpoint_group_size: int = 0,
    learning_rate: float = 4.0e-5,
) -> SingleStepSupervised | TemporalRolloutSupervised:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = PDETransformer(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        type=model_type,
        patch_size=patch_size,
        periodic=periodic,
        carrier_token_active=carrier_token_active,
        use_ttt_window_attention=use_ttt_window_attention,
        token_mixer_type=token_mixer_type,
        ttt_layer_type=ttt_layer_type,
        ttt_mini_batch_size=ttt_mini_batch_size,
        ttt_base_lr=ttt_base_lr,
        ttt_use_gate=ttt_use_gate,
        ttt_scan_checkpoint_group_size=ttt_scan_checkpoint_group_size,
        vittt_inner_lr=vittt_inner_lr,
        vittt_padding_mode=vittt_padding_mode,
        attention_ttt_type=attention_ttt_type,
        attention_ttt_gate_init=attention_ttt_gate_init,
        attention_ttt_bidirectional=attention_ttt_bidirectional,
        global_ttt_stage_names=global_ttt_stage_names,
        global_ttt_inner_lr=global_ttt_inner_lr,
        global_ttt_gate_init=global_ttt_gate_init,
        global_ttt_key_norm=global_ttt_key_norm,
        temporal_ttt_enabled=temporal_ttt_enabled,
        temporal_ttt_layer_type=temporal_ttt_layer_type,
        temporal_ttt_mini_batch_size=temporal_ttt_mini_batch_size,
        temporal_ttt_base_lr=temporal_ttt_base_lr,
        temporal_ttt_gate_init=temporal_ttt_gate_init,
        temporal_ttt_use_output_gate=temporal_ttt_use_output_gate,
        temporal_ttt_scan_checkpoint_group_size=temporal_ttt_scan_checkpoint_group_size,
    )
    if training_mode == "temporal_rollout":
        strategy = TemporalRolloutSupervised(
            model=model,
            image_key=0,
            optimizer="adamw",
            train_unrolling_steps=train_unrolling_steps,
            tbptt_chunk_size=tbptt_chunk_size,
            gradient_accumulation_batches=gradient_accumulation_batches,
            temporal_ttt_learning_rate=temporal_ttt_learning_rate,
        )
    elif training_mode == "single_step":
        strategy = SingleStepSupervised(
            model=model,
            image_key=0,
            optimizer="adamw",
            use_ttt_state_cache_inference=use_ttt_state_cache_inference,
            use_ttt_state_cache_train=use_ttt_state_cache_train,
        )
    else:
        raise ValueError(f"Unsupported training_mode: {training_mode}")
    strategy.learning_rate = learning_rate
    return strategy


def build_trainer(args: argparse.Namespace, run_root: Path) -> L.Trainer:
    checkpoint_monitor = "val/loss" if args.training_mode == "temporal_rollout" else "val/loss_epoch"
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_root / args.run_name / "checkpoints",
        filename="epoch-{epoch:03d}",
        monitor=checkpoint_monitor,
        mode="min",
        save_last=True,
        save_top_k=3,
        every_n_epochs=1,
    )
    trainer_strategy = args.strategy
    if args.training_mode == "temporal_rollout" and trainer_strategy == "ddp":
        trainer_strategy = "ddp_find_unused_parameters_true"
    return L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        strategy=trainer_strategy,
        precision=args.precision,
        accumulate_grad_batches=1 if args.training_mode == "temporal_rollout" else args.accumulate_grad_batches,
        callbacks=[checkpoint_callback, EpochPrintCallback()],
        logger=CSVLogger(save_dir=str(run_root), name=args.run_name),
        enable_progress_bar=False,
        log_every_n_steps=10,
    )


def initialize_matching_weights(strategy: SingleStepSupervised, checkpoint_path: Path) -> None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    print("initializing matching weights from:", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = strategy.load_state_dict(state_dict, strict=False)
    new_module_markers = (".global_ttt.", ".temporal_ttt.")
    invalid_missing = [key for key in missing if not any(marker in key for marker in new_module_markers)]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Initialization checkpoint does not match the base architecture: "
            f"invalid_missing={invalid_missing[:5]} unexpected={unexpected[:5]}"
        )
    print(f"initialized base model; new global TTT parameters: {len(missing)}")


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


def print_checkpoint_listing(run_root: Path, run_name: str) -> None:
    candidates = [
        run_root / run_name / "checkpoints",
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
    resolved_token_mixer_type = resolve_token_mixer_type(args.token_mixer_type, args.use_ttt_window_attention)
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
    print("run_name:", args.run_name)
    print("model_type:", args.model_type)
    print("in_channels:", args.in_channels, "out_channels:", args.out_channels, "patch_size:", args.patch_size)
    print("periodic:", args.periodic, "carrier_token_active:", args.carrier_token_active)
    print("checkpoint_path:", checkpoint_path)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
    print("precision:", args.precision)
    print("devices:", args.devices, "strategy:", args.strategy)
    print("accumulate_grad_batches:", args.accumulate_grad_batches)
    print("use_ttt_window_attention:", args.use_ttt_window_attention)
    print("token_mixer_type:", args.token_mixer_type, "resolved:", resolved_token_mixer_type)
    print("use_ttt_state_cache_train:", args.use_ttt_state_cache_train)
    print("ttt_layer_type:", args.ttt_layer_type)
    print("ttt_mini_batch_size:", args.ttt_mini_batch_size)
    print("ttt_base_lr:", args.ttt_base_lr)
    print("ttt_use_gate:", args.ttt_use_gate)
    print("ttt_scan_checkpoint_group_size:", args.ttt_scan_checkpoint_group_size)
    print("vittt_inner_lr:", args.vittt_inner_lr)
    print("vittt_padding_mode:", args.vittt_padding_mode)
    print("attention_ttt_type:", args.attention_ttt_type)
    print("attention_ttt_gate_init:", args.attention_ttt_gate_init)
    print("attention_ttt_bidirectional:", args.attention_ttt_bidirectional)
    print("global_ttt_stage_names:", args.global_ttt_stage_names)
    print("global_ttt_inner_lr:", args.global_ttt_inner_lr)
    print("global_ttt_gate_init:", args.global_ttt_gate_init)
    print("global_ttt_key_norm:", args.global_ttt_key_norm)
    print("training_mode:", args.training_mode)
    print("train_unrolling_steps:", args.train_unrolling_steps)
    print("train_step_size:", args.train_step_size)
    print("tbptt_chunk_size:", args.tbptt_chunk_size)
    print("temporal_ttt_enabled:", args.temporal_ttt_enabled)
    print("temporal_ttt_layer_type:", args.temporal_ttt_layer_type)
    print("temporal_ttt_mini_batch_size:", args.temporal_ttt_mini_batch_size)
    print("temporal_ttt_base_lr:", args.temporal_ttt_base_lr)
    print("temporal_ttt_gate_init:", args.temporal_ttt_gate_init)
    print("temporal_ttt_learning_rate:", args.temporal_ttt_learning_rate)
    print("learning_rate:", args.learning_rate)
    print("generate_data:", args.generate_data, "low_res:", args.low_res)
    print(
        "downsample_factor:", args.downsample_factor,
        "sample_size:", args.sample_size,
        "test_unrolling_steps:", args.test_unrolling_steps,
        "max_channels:", args.max_channels,
    )

    run_root.mkdir(parents=True, exist_ok=True)
    if args.generate_data:
        generate_ape_xxl_data(args, data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if any(
        v is not None
        for v in (
            args.sims_split_joint_train,
            args.sims_split_joint_test,
            args.sims_split_gs_train,
            args.sims_split_gs_test,
        )
    ):
        apply_sims_split_override(
            joint_train=args.sims_split_joint_train,
            joint_test=args.sims_split_joint_test,
            gs_train=args.sims_split_gs_train,
            gs_test=args.sims_split_gs_test,
        )

    data_module = build_data_module(
        data_dir, FULL_DATASET_NAMES, args.batch_size, args.num_workers,
        downsample_factor=args.downsample_factor,
        train_unrolling_steps=args.train_unrolling_steps,
        train_step_size=args.train_step_size,
        test_unrolling_steps=args.test_unrolling_steps,
        max_channels=args.max_channels,
    )

    if args.training_mode == "temporal_rollout":
        if not args.temporal_ttt_enabled:
            raise SystemExit("training_mode=temporal_rollout requires temporal_ttt_enabled=true")
        if args.train_unrolling_steps < 2:
            raise SystemExit("temporal rollout training requires train_unrolling_steps >= 2")
        if args.use_ttt_state_cache_train:
            raise SystemExit(
                "temporal rollout owns its persistent state; set the legacy "
                "use_ttt_state_cache_train=false"
            )
    elif args.train_unrolling_steps != 1:
        raise SystemExit("single_step training requires train_unrolling_steps=1")

    if args.use_ttt_state_cache_train and resolved_token_mixer_type == "attention":
        raise SystemExit(
            "use_ttt_state_cache_train=true requires use_ttt_window_attention=true "
            "(the plain baseline has no TTT state to cache)."
        )
    if args.use_ttt_state_cache_train and resolved_token_mixer_type in ("vittt", "attention_ttt"):
        raise SystemExit(
            f"use_ttt_state_cache_train=true is invalid for token_mixer_type={resolved_token_mixer_type}. "
            "This experiment has no cross-step TTT state cache; set use_ttt_state_cache_train=false."
        )

    strategy = build_strategy(
        args.seed,
        model_type=args.model_type,
        use_ttt_state_cache_inference=False,
        use_ttt_state_cache_train=args.use_ttt_state_cache_train,
        sample_size=args.sample_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        patch_size=args.patch_size,
        periodic=args.periodic,
        carrier_token_active=args.carrier_token_active,
        use_ttt_window_attention=args.use_ttt_window_attention,
        token_mixer_type=args.token_mixer_type,
        ttt_layer_type=args.ttt_layer_type,
        ttt_mini_batch_size=args.ttt_mini_batch_size,
        ttt_base_lr=args.ttt_base_lr,
        ttt_use_gate=args.ttt_use_gate,
        ttt_scan_checkpoint_group_size=args.ttt_scan_checkpoint_group_size,
        vittt_inner_lr=args.vittt_inner_lr,
        vittt_padding_mode=args.vittt_padding_mode,
        attention_ttt_type=args.attention_ttt_type,
        attention_ttt_gate_init=args.attention_ttt_gate_init,
        attention_ttt_bidirectional=args.attention_ttt_bidirectional,
        global_ttt_stage_names=args.global_ttt_stage_names,
        global_ttt_inner_lr=args.global_ttt_inner_lr,
        global_ttt_gate_init=args.global_ttt_gate_init,
        global_ttt_key_norm=args.global_ttt_key_norm,
        training_mode=args.training_mode,
        train_unrolling_steps=args.train_unrolling_steps,
        tbptt_chunk_size=args.tbptt_chunk_size,
        gradient_accumulation_batches=args.accumulate_grad_batches,
        temporal_ttt_enabled=args.temporal_ttt_enabled,
        temporal_ttt_layer_type=args.temporal_ttt_layer_type,
        temporal_ttt_mini_batch_size=args.temporal_ttt_mini_batch_size,
        temporal_ttt_base_lr=args.temporal_ttt_base_lr,
        temporal_ttt_gate_init=args.temporal_ttt_gate_init,
        temporal_ttt_learning_rate=args.temporal_ttt_learning_rate,
        temporal_ttt_use_output_gate=args.temporal_ttt_use_output_gate,
        temporal_ttt_scan_checkpoint_group_size=args.temporal_ttt_scan_checkpoint_group_size,
        learning_rate=args.learning_rate,
    )
    trainer = build_trainer(args, run_root)

    should_resume = args.resume or (args.auto_resume and checkpoint_path.exists())
    ckpt_path = str(checkpoint_path) if should_resume else None
    if not should_resume and args.init_checkpoint_path is not None:
        initialize_matching_weights(strategy, args.init_checkpoint_path)
    print("resuming from checkpoint:" if ckpt_path else "starting fresh training run", ckpt_path or "")

    trainer.fit(strategy, datamodule=data_module, ckpt_path=ckpt_path)
    val_metrics = trainer.validate(strategy, datamodule=data_module, verbose=False)

    checkpoint_callback = trainer.checkpoint_callback
    print("last checkpoint:", checkpoint_callback.last_model_path)
    print("best checkpoints:", checkpoint_callback.best_k_models)
    print("validation:", val_metrics)
    print_checkpoint_listing(run_root, args.run_name)

    if args.skip_rollout_eval:
        return

    experiment_results = []
    if args.temporal_ttt_enabled:
        eval_modes = [("inference_persistent_temporal_state", True)]
    else:
        eval_modes = [("inference_cache_off", False)]
    if resolved_token_mixer_type == "ttt_sequence" and not args.temporal_ttt_enabled:
        eval_modes.append(("inference_cache_on", True))
    for name, use_cache in eval_modes:
        print(f"\n=== Evaluating {name} ===")
        result = {
            "name": name,
            "checkpoint": trainer.checkpoint_callback.last_model_path,
            "max_epochs": args.max_epochs,
            "val_metrics": val_metrics,
            **run_rollout_check(strategy, data_module, use_cache, num_frames=args.test_unrolling_steps),
        }
        experiment_results.append(result)
        print(json.dumps(result, indent=2))

    off = next((r for r in experiment_results if not r["use_ttt_state_cache_inference"]), None)
    on = next((r for r in experiment_results if r["use_ttt_state_cache_inference"]), None)
    if off is not None:
        print("cache off rollout mse:", off["rollout_mse"])
    if on is not None:
        label = "persistent temporal state" if args.temporal_ttt_enabled else "cache on"
        print(f"{label} rollout mse:", on["rollout_mse"])
    if off is not None and on is not None:
        print("prediction shapes:", off["prediction_shape"], on["prediction_shape"])
    elif off is not None:
        print("prediction shape:", off["prediction_shape"])
    elif on is not None:
        print("prediction shape:", on["prediction_shape"])

    del strategy, data_module
    gc.collect()


if __name__ == "__main__":
    main()
