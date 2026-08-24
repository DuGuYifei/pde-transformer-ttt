"""Official-data evaluator for PDE-Transformer models.

This script is intentionally self-contained: it imports the installed
``pdetransformer`` package directly and can be copied to the server together with
one YAML config. It supports both model sources used in this project:

* local Lightning ``.ckpt`` files trained from ``server_example/*.yaml``;
* official diffusers/safetensors checkpoints loaded with
  ``PDETransformer.from_pretrained``.

The output schema is compatible with the previous official evaluation scripts:
``results_cache_off.csv/json``, optional ``results_cache_on.csv/json`` for the
old sequence-style TTT cache, and ``summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

# Prefer this reviewed source tree over an older installed wheel on the server.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
for package_name, package_path in {
    "pdetransformer": REPO_ROOT / "pdetransformer",
    "pdetransformer.core": REPO_ROOT / "pdetransformer" / "core",
}.items():
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

from pdetransformer.core.mixed_channels import PDETransformer, SingleStepSupervised
from pdetransformer.data import MultiDataModule
from pdetransformer.data.pbdl_datatypes.ape_2d_splits import (
    DATASET_PROFILES,
    SEPARATE_TEST_DATASETS,
    ape_2d_xxl_simulation_split,
    dataset_names_for_profile,
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

CONFIG_DEFAULTS: dict[str, Any] = {
    "work_dir": "~/working",
    "data_dir": None,
    "run_root": "~/working/runs_v2",
    "run_name": "pretrained_eval",
    "model_type": "PDE-S",
    "in_channels": 2,
    "out_channels": 2,
    "patch_size": 4,
    "periodic": True,
    "carrier_token_active": False,
    "token_mixer_type": None,
    "use_ttt_window_attention": False,
    "use_ttt_state_cache_train": False,
    "ttt_layer_type": "linear",
    "ttt_mini_batch_size": 16,
    "ttt_base_lr": 1.0,
    "ttt_use_gate": False,
    "ttt_scan_checkpoint_group_size": 0,
    "vittt_inner_lr": 1.0,
    "vittt_head_dim": 32,
    "vittt_padding_mode": "zero",
    "attention_ttt_type": "ttt_sequence",
    "attention_ttt_gate_init": 0.1,
    "attention_ttt_bidirectional": True,
    "global_ttt_stage_names": [],
    "global_ttt_inner_lr": 1.0,
    "global_ttt_gate_init": 0.0,
    "global_ttt_key_norm": True,
    "batch_size": 8,
    "num_workers": 2,
    "seed": 42,
    "downsample_factor": 2,
    "sample_size": 128,
    "test_unrolling_steps": 29,
    "max_channels": 2,
    "checkpoint_path": None,
    "dataset_profile": "legacy_small",
    "strict_test_split": False,
    "legacy_all_source_sims": False,
    "id_ood_test": False,
}

DEFAULT_EVAL_K = (1, 10, 20, 29)
DEFAULT_ROLLOUT_STEPS = 30
ID_OOD_CONDITIONS = ("id", "ood_low", "ood_high")
ID_OOD_EXPECTED_SIM_IDS = list(range(9))
ID_OOD_EXPECTED_FRAMES = 30


def _path_or_none(value: Any) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    return Path(value)


def _load_config(path: Path | None) -> dict[str, Any]:
    cfg = dict(CONFIG_DEFAULTS)
    if path is None:
        return cfg

    from omegaconf import OmegaConf

    path = Path(path).expanduser()
    if not path.exists():
        raise SystemExit(f"Config file does not exist: {path}")
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config file must contain a mapping: {path}")
    cfg.update(loaded)
    return cfg


def _format_million(n: int) -> str:
    return f"{n / 1e6:.2f}M"


def _format_int(n: int) -> str:
    return f"{n:,}"


def _resolve_token_mixer(token_mixer_type: str | None, use_ttt_window_attention: bool) -> str:
    if token_mixer_type is not None:
        return token_mixer_type
    return "ttt_sequence" if use_ttt_window_attention else "attention"


def _expand(path: Path | None) -> Path | None:
    return path.expanduser().resolve() if path is not None else None


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional server_example YAML. All model/data eval keys are read.",
    )
    config_args, remaining = config_parser.parse_known_args()
    cfg = _load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a local PDE-Transformer checkpoint or official "
            "from_pretrained safetensors model on the official/server data."
        ),
        parents=[config_parser],
    )
    parser.set_defaults(config=config_args.config)

    parser.add_argument("--work-dir", type=Path, default=_path_or_none(cfg["work_dir"]))
    parser.add_argument("--data-dir", type=Path, default=_path_or_none(cfg["data_dir"]))
    parser.add_argument("--run-root", type=Path, default=_path_or_none(cfg["run_root"]))
    parser.add_argument("--run-name", type=str, default=cfg.get("run_name", "pretrained_eval"))

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=_path_or_none(cfg.get("checkpoint_path")),
        help=(
            "Lightning .ckpt to evaluate. If omitted, the script evaluates the "
            "official/from_pretrained model selected by --model-source."
        ),
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default="thuerey-group/pde-transformer",
        help=(
            "HuggingFace repo id or local diffusers directory for official "
            "safetensors/config.json weights."
        ),
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="mc-s",
        help=(
            "Subfolder for from_pretrained, e.g. mc-s. Use empty string if "
            "--model-source already points at the leaf folder."
        ),
    )

    parser.add_argument("--model-type", choices=("PDE-S", "PDE-B", "PDE-L"), default=cfg["model_type"])
    parser.add_argument("--in-channels", type=int, default=cfg["in_channels"])
    parser.add_argument("--out-channels", type=int, default=cfg["out_channels"])
    parser.add_argument("--patch-size", type=int, default=cfg["patch_size"])
    parser.add_argument("--periodic", action=argparse.BooleanOptionalAction, default=cfg["periodic"])
    parser.add_argument(
        "--carrier-token-active",
        action=argparse.BooleanOptionalAction,
        default=cfg["carrier_token_active"],
    )
    parser.add_argument(
        "--token-mixer-type",
        choices=(
            "attention",
            "global_vittt",
            "global_h_vittt",
            "global_linear_ttt",
            "window_linear_ttt",
            "window_fullbatch_mlp_ttt",
        ),
        default=cfg.get("token_mixer_type"),
        help="Local checkpoint mixer type. Ignored for from_pretrained models.",
    )
    parser.add_argument(
        "--use-ttt-window-attention",
        action=argparse.BooleanOptionalAction,
        default=cfg["use_ttt_window_attention"],
        help="Legacy flag: true maps to ttt_sequence when token_mixer_type is unset.",
    )
    parser.add_argument(
        "--use-ttt-state-cache-train",
        action=argparse.BooleanOptionalAction,
        default=cfg["use_ttt_state_cache_train"],
        help="Recorded training cache flag; inference cache is controlled by --cache-mode.",
    )
    parser.add_argument("--ttt-layer-type", choices=("linear", "mlp"), default=cfg["ttt_layer_type"])
    parser.add_argument("--ttt-mini-batch-size", type=int, default=cfg["ttt_mini_batch_size"])
    parser.add_argument("--ttt-base-lr", type=float, default=cfg["ttt_base_lr"])
    parser.add_argument("--ttt-use-gate", action=argparse.BooleanOptionalAction, default=cfg["ttt_use_gate"])
    parser.add_argument(
        "--ttt-scan-checkpoint-group-size",
        type=int,
        default=cfg["ttt_scan_checkpoint_group_size"],
    )
    parser.add_argument("--vittt-inner-lr", type=float, default=cfg["vittt_inner_lr"])
    parser.add_argument("--vittt-head-dim", type=int, default=cfg["vittt_head_dim"])
    parser.add_argument(
        "--vittt-padding-mode",
        choices=("zero", "replicate"),
        default=cfg["vittt_padding_mode"],
    )
    parser.add_argument(
        "--attention-ttt-type",
        choices=("ttt_sequence", "vittt"),
        default=cfg["attention_ttt_type"],
        help="Post-attention TTT branch used when token_mixer_type=attention_ttt.",
    )
    parser.add_argument("--attention-ttt-gate-init", type=float, default=cfg["attention_ttt_gate_init"])
    parser.add_argument(
        "--attention-ttt-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=cfg["attention_ttt_bidirectional"],
    )
    parser.add_argument("--global-ttt-inner-lr", type=float, default=cfg["global_ttt_inner_lr"])
    parser.add_argument("--global-ttt-gate-init", type=float, default=cfg["global_ttt_gate_init"])
    parser.add_argument(
        "--global-ttt-key-norm",
        action=argparse.BooleanOptionalAction,
        default=cfg["global_ttt_key_norm"],
    )

    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--downsample-factor", type=int, default=cfg["downsample_factor"])
    parser.add_argument("--sample-size", type=int, default=cfg["sample_size"])
    parser.add_argument("--test-unrolling-steps", type=int, default=cfg["test_unrolling_steps"])
    parser.add_argument("--max-channels", type=int, default=cfg["max_channels"])
    parser.add_argument(
        "--dataset-profile",
        choices=tuple(sorted(DATASET_PROFILES)),
        default=cfg["dataset_profile"],
        help="Select the reviewed legacy_small or full_paper simulation split.",
    )

    parser.add_argument(
        "--cache-mode",
        choices=("auto", "off", "on", "both"),
        default="auto",
        help=(
            "TTT state cache mode during inference. auto means both for "
            "ttt_sequence checkpoints, off for attention/vittt/attention_ttt/pretrained."
        ),
    )
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--eval-k", type=int, nargs="+", default=list(DEFAULT_EVAL_K))
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Defaults to the datasets defined by --dataset-profile.",
    )
    parser.add_argument(
        "--strict-test-split",
        action=argparse.BooleanOptionalAction,
        default=cfg["strict_test_split"],
        help="Validate source files, simulation IDs, and one-rollout-per-simulation.",
    )
    parser.add_argument(
        "--legacy-all-source-sims",
        action=argparse.BooleanOptionalAction,
        default=cfg["legacy_all_source_sims"],
        help=(
            "Reproduce the historical 128-resolution architecture-screen protocol "
            "by evaluating every simulation in each selected source file."
        ),
    )
    parser.add_argument(
        "--id-ood-test",
        action=argparse.BooleanOptionalAction,
        default=cfg["id_ood_test"],
        help=(
            "Evaluate datasets_test/<pde>.hdf5 using sim0..8 exactly once: "
            "ID sim0..2, OOD-low sim3..5, and OOD-high sim6..8."
        ),
    )
    parser.add_argument("--max-batches-per-dataset", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-nonstrict-load",
        action="store_true",
        help="Do not fail on missing/unexpected checkpoint keys.",
    )

    parser.set_defaults(global_ttt_stage_names=cfg["global_ttt_stage_names"])
    args = parser.parse_args(remaining)
    if args.strict_test_split and args.legacy_all_source_sims:
        parser.error("--strict-test-split and --legacy-all-source-sims are mutually exclusive")
    if args.id_ood_test and args.legacy_all_source_sims:
        parser.error("--id-ood-test and --legacy-all-source-sims are mutually exclusive")
    if args.datasets is None:
        args.datasets = list(dataset_names_for_profile(args.dataset_profile))
    return args


def build_data_module(
    data_dir: Path,
    dataset_names: list[str],
    batch_size: int,
    num_workers: int,
    downsample_factor: int,
    test_unrolling_steps: int,
    max_channels: int,
    dataset_profile: str,
    test_sim_ids_override: list[int] | None = None,
) -> MultiDataModule:
    params_data = {
        "path_index": {"2D_APE_xxl": str(data_dir)},
        "dataset_names": dataset_names,
        "dataset_type": "2D_APE_xxl",
        "unrolling_steps": 1,
        "test_unrolling_steps": test_unrolling_steps,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "cache_strategy": "none",
        "different_resolution_strategy": "none",
        "normalize_data": "mean-std",
        "normalize_const": "mean-std",
        "downsample_factor": downsample_factor,
        "max_channels": max_channels,
        "dataset_profile": dataset_profile,
    }
    if test_sim_ids_override is not None:
        params_data["test_sim_ids_override"] = list(test_sim_ids_override)
    return MultiDataModule(**params_data)


def all_source_sim_ids(data_dir: Path, pde: str) -> list[int]:
    source_name = pde + "_test" if pde in SEPARATE_TEST_DATASETS else pde
    source_file = data_dir / f"{source_name}.hdf5"
    if not source_file.is_file():
        raise FileNotFoundError(f"Historical evaluation source not found: {source_file}")
    with h5py.File(source_file, "r") as handle:
        if "sims" not in handle:
            raise RuntimeError(f"Historical evaluation source has no 'sims' group: {source_file}")
        sim_ids = sorted(int(name.removeprefix("sim")) for name in handle["sims"].keys())
    if sim_ids != list(range(len(sim_ids))):
        raise RuntimeError(f"Simulation IDs are not contiguous in {source_file}: {sim_ids}")
    return sim_ids


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def per_trajectory_nrmse(pred_k: np.ndarray, ref_k: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, pred_k.ndim))
    mse_pred = np.mean((pred_k - ref_k) ** 2, axis=axes)
    mse_zero = np.mean(ref_k ** 2, axis=axes)
    return np.sqrt(mse_pred / np.clip(mse_zero, 1e-30, None))


def inspect_first_batch(loader) -> dict[str, Any]:
    batch = next(iter(loader))
    data = batch["data"] if isinstance(batch, dict) else batch[0]
    if data.ndim != 5:
        raise RuntimeError(f"Unexpected data shape {tuple(data.shape)}; expected (B, T, C, H, W).")
    _, t, c, h, w = data.shape
    return {"trajectory_length": int(t), "data_shape": [int(c), int(h), int(w)], "channels": int(c)}


def inspect_test_split(dm: MultiDataModule, pde: str) -> dict[str, Any]:
    if len(dm.subsets_test) != 1:
        raise RuntimeError(f"Expected one test subset for {pde}, found {len(dm.subsets_test)}.")

    raw_dataset = dm.subsets_test[0].dataset
    selected_sim_ids = (
        list(raw_dataset.sel_sims)
        if raw_dataset.sel_sims is not None
        else list(range(raw_dataset.num_sims))
    )
    return {
        "requested_dataset_name": pde,
        "source_dataset_name": raw_dataset.dset_name,
        "source_file": str(Path(raw_dataset.dset_file).resolve()),
        "source_file_name": Path(raw_dataset.dset_file).name,
        "source_num_simulations": int(raw_dataset.num_sims),
        "source_num_frames": int(raw_dataset.num_frames),
        "selected_sim_ids": [int(sim_id) for sim_id in selected_sim_ids],
        "selected_num_simulations": len(selected_sim_ids),
        "samples_per_simulation": int(raw_dataset.samples_per_sim),
    }


def validate_profile_test_split(
    pde: str,
    dataset_profile: str,
    split_info: dict[str, Any],
) -> None:
    train_sims, test_sims = ape_2d_xxl_simulation_split(pde, dataset_profile)
    expected_source = pde + "_test" if pde in SEPARATE_TEST_DATASETS else pde
    expected_sim_ids = (
        list(range(split_info["source_num_simulations"]))
        if test_sims is None
        else list(test_sims)
    )
    if split_info["source_dataset_name"] != expected_source:
        raise RuntimeError(
            f"Strict test for {pde} loaded {split_info['source_dataset_name']!r}; "
            f"expected {expected_source!r}."
        )
    if split_info["source_file_name"] != f"{expected_source}.hdf5":
        raise RuntimeError(
            f"Strict test for {pde} loaded {split_info['source_file_name']!r}; "
            f"expected {expected_source}.hdf5."
        )
    if split_info["selected_sim_ids"] != expected_sim_ids:
        raise RuntimeError(
            f"Strict test for {pde} selected {split_info['selected_sim_ids']}; "
            f"expected {expected_sim_ids}."
        )
    if split_info["samples_per_simulation"] != 1:
        raise RuntimeError(
            f"Strict test for {pde} produced "
            f"{split_info['samples_per_simulation']} samples per simulation; "
            "expected one 29-step rollout."
        )
    if train_sims is not None and set(train_sims) & set(expected_sim_ids):
        raise RuntimeError(f"Train/test simulation overlap detected for {pde}.")


def load_id_ood_manifest(
    data_dir: Path,
    datasets: list[str],
    sample_size: int,
    downsample_factor: int,
) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"ID/OOD test manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_scalars = {
        "solver_resolution": 2048,
        "stored_resolution": 256,
        "time_steps": 30,
        "rollout_transitions": 29,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"ID/OOD manifest has {key}={manifest.get(key)!r}; expected {expected!r}."
            )
    stored_resolution = int(manifest["stored_resolution"])
    if stored_resolution % downsample_factor != 0:
        raise RuntimeError(
            f"Stored resolution {stored_resolution} is not divisible by "
            f"downsample_factor={downsample_factor}."
        )
    actual_resolution = stored_resolution // downsample_factor
    if actual_resolution != sample_size:
        raise RuntimeError(
            f"ID/OOD data resolves to {actual_resolution}, but model sample_size={sample_size}."
        )
    if tuple(manifest.get("conditions", ())) != ID_OOD_CONDITIONS:
        raise RuntimeError(
            f"ID/OOD manifest conditions are {manifest.get('conditions')!r}; "
            f"expected {list(ID_OOD_CONDITIONS)!r}."
        )

    manifest_pdes = manifest.get("pdes")
    if not isinstance(manifest_pdes, dict):
        raise RuntimeError("ID/OOD manifest is missing the 'pdes' mapping.")
    missing = sorted(set(datasets) - set(manifest_pdes))
    if missing:
        raise RuntimeError(f"ID/OOD manifest is missing requested PDEs: {missing}.")

    expected_seeds = list(manifest.get("seeds", ()))
    if len(expected_seeds) != 3 or len(set(expected_seeds)) != 3:
        raise RuntimeError(f"ID/OOD manifest must define three unique seeds; got {expected_seeds}.")

    for pde in datasets:
        entries = manifest_pdes[pde]
        if not isinstance(entries, list) or len(entries) != 9:
            raise RuntimeError(f"ID/OOD manifest must define 9 entries for {pde}.")
        entries_by_sim = {int(entry["sim_id"]): entry for entry in entries}
        if sorted(entries_by_sim) != ID_OOD_EXPECTED_SIM_IDS:
            raise RuntimeError(
                f"ID/OOD manifest sim IDs for {pde} are {sorted(entries_by_sim)}; "
                f"expected {ID_OOD_EXPECTED_SIM_IDS}."
            )
        for condition in ID_OOD_CONDITIONS:
            condition_entries = sorted(
                (entry for entry in entries if entry.get("condition") == condition),
                key=lambda entry: int(entry["sim_id"]),
            )
            if len(condition_entries) != 3:
                raise RuntimeError(f"ID/OOD manifest must define 3 {condition} entries for {pde}.")
            if [entry.get("seed") for entry in condition_entries] != expected_seeds:
                raise RuntimeError(
                    f"ID/OOD seed order for {pde}/{condition} does not match manifest seeds."
                )
        if not (data_dir / f"{pde}.hdf5").exists():
            raise RuntimeError(f"ID/OOD data file not found: {data_dir / f'{pde}.hdf5'}")
    return manifest


def validate_id_ood_test_split(
    pde: str,
    split_info: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if split_info["source_dataset_name"] != pde:
        raise RuntimeError(
            f"ID/OOD test for {pde} loaded {split_info['source_dataset_name']!r}; "
            f"expected {pde!r}."
        )
    if split_info["source_file_name"] != f"{pde}.hdf5":
        raise RuntimeError(
            f"ID/OOD test for {pde} loaded {split_info['source_file_name']!r}; "
            f"expected {pde}.hdf5."
        )
    if split_info["source_num_frames"] != ID_OOD_EXPECTED_FRAMES:
        raise RuntimeError(
            f"ID/OOD test for {pde} found {split_info['source_num_frames']} frames; "
            f"expected {ID_OOD_EXPECTED_FRAMES}."
        )
    if split_info["selected_sim_ids"] != ID_OOD_EXPECTED_SIM_IDS:
        raise RuntimeError(
            f"ID/OOD test for {pde} selected {split_info['selected_sim_ids']}; "
            f"expected {ID_OOD_EXPECTED_SIM_IDS}."
        )
    if split_info["samples_per_simulation"] != 1:
        raise RuntimeError(
            f"ID/OOD test for {pde} produced "
            f"{split_info['samples_per_simulation']} samples per simulation; "
            "expected one 29-step rollout."
        )
    manifest_ids = sorted(int(entry["sim_id"]) for entry in manifest["pdes"][pde])
    if manifest_ids != split_info["selected_sim_ids"]:
        raise RuntimeError(f"ID/OOD test split for {pde} does not match manifest sim IDs.")


def build_checkpoint_strategy(args: argparse.Namespace, checkpoint_path: Path) -> SingleStepSupervised:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = PDETransformer(
        sample_size=args.sample_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        type=args.model_type,
        patch_size=args.patch_size,
        periodic=args.periodic,
        carrier_token_active=args.carrier_token_active,
        token_mixer_type=args.token_mixer_type,
        vittt_inner_lr=args.vittt_inner_lr,
        vittt_head_dim=args.vittt_head_dim,
    )
    strategy = SingleStepSupervised(
        model=model,
        image_key=0,
        optimizer="adamw",
    )
    strategy.learning_rate = 0.0

    print(f"loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        raise SystemExit(f"Unsupported checkpoint format: {checkpoint_path}")
    missing, unexpected = strategy.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} first={missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} first={unexpected[:5]}")
    if (missing or unexpected) and not args.allow_nonstrict_load:
        raise SystemExit(
            "Checkpoint did not match the constructed model. Check --config, "
            "--token-mixer-type, --ttt-layer-type, and ViTTT-style args, or pass "
            "--allow-nonstrict-load only for debugging."
        )
    return strategy


def build_pretrained_strategy(args: argparse.Namespace) -> SingleStepSupervised:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    load_kwargs: dict[str, Any] = {}
    if args.subfolder:
        load_kwargs["subfolder"] = args.subfolder
    print(f"loading from_pretrained: source={args.model_source!r} subfolder={args.subfolder!r}")
    model = PDETransformer.from_pretrained(args.model_source, **load_kwargs)
    strategy = SingleStepSupervised(
        model=model,
        image_key=0,
        optimizer="adamw",
    )
    strategy.learning_rate = 0.0
    return strategy


def evaluate_dataset(
    strategy: SingleStepSupervised,
    args: argparse.Namespace,
    data_dir: Path,
    pde: str,
    device: torch.device,
) -> dict[str, Any]:
    if args.id_ood_test:
        test_sim_ids_override = ID_OOD_EXPECTED_SIM_IDS
    elif args.legacy_all_source_sims:
        # The standard loader already selects every trajectory from dedicated
        # *_test.hdf5 sources. An explicit override would incorrectly redirect
        # those PDEs back to their training file.
        test_sim_ids_override = (
            None if pde in SEPARATE_TEST_DATASETS else all_source_sim_ids(data_dir, pde)
        )
    else:
        test_sim_ids_override = None

    dm = build_data_module(
        data_dir=data_dir,
        dataset_names=[pde],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        downsample_factor=args.downsample_factor,
        test_unrolling_steps=args.test_unrolling_steps,
        max_channels=args.max_channels,
        dataset_profile=args.dataset_profile,
        test_sim_ids_override=test_sim_ids_override,
    )
    dm.setup(stage="test")
    split_info = inspect_test_split(dm, pde)
    if args.strict_test_split and not args.id_ood_test:
        validate_profile_test_split(pde, args.dataset_profile, split_info)
    if args.id_ood_test:
        validate_id_ood_test_split(pde, split_info, args.id_ood_manifest)

    loader = dm.test_dataloader()
    num_trajectories = len(dm.set_test) if dm.set_test is not None else 0
    expected_trajectories = (
        split_info["selected_num_simulations"] * split_info["samples_per_simulation"]
    )
    if num_trajectories != expected_trajectories:
        raise RuntimeError(
            f"Test dataset {pde} has {num_trajectories} samples, but split provenance "
            f"implies {expected_trajectories}."
        )

    info_loader = dm.test_dataloader()
    shape_info = inspect_first_batch(info_loader)
    del info_loader

    sum_nrmse = {k: 0.0 for k in args.eval_k}
    trajectory_nrmse = {k: [] for k in args.eval_k}
    count = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches_per_dataset is not None and batch_idx >= args.max_batches_per_dataset:
                break

            prediction, reference = strategy.predict(batch, device=device, num_frames=args.rollout_steps)
            prediction = np.asarray(prediction)
            reference = np.asarray(reference)

            batch_b = prediction.shape[0]
            for k in args.eval_k:
                per_traj = per_trajectory_nrmse(prediction[:, k], reference[:, k])
                sum_nrmse[k] += float(per_traj.sum())
                trajectory_nrmse[k].extend(float(value) for value in per_traj)
            count += batch_b

    elapsed = time.perf_counter() - t0
    nrmse = {
        k: (sum_nrmse[k] / count if count > 0 else float("nan"))
        for k in args.eval_k
    }
    trajectory_results: list[dict[str, Any]] = []
    condition_nrmse: dict[str, dict[int, float]] = {}
    condition_sum_nrmse: dict[str, dict[int, float]] = {}
    condition_counts: dict[str, int] = {}
    if args.id_ood_test:
        selected_sim_ids = split_info["selected_sim_ids"][:count]
        entries_by_sim = {
            int(entry["sim_id"]): entry
            for entry in args.id_ood_manifest["pdes"][pde]
        }
        for trajectory_idx, sim_id in enumerate(selected_sim_ids):
            entry = entries_by_sim[sim_id]
            trajectory_results.append(
                {
                    "sim_id": sim_id,
                    "condition": entry["condition"],
                    "seed": int(entry["seed"]),
                    "parameter_overrides": entry.get("parameter_overrides", {}),
                    "numerical_overrides": entry.get("numerical_overrides", {}),
                    **{
                        f"nRMSE_{k}": trajectory_nrmse[k][trajectory_idx]
                        for k in args.eval_k
                    },
                }
            )
        for condition in ID_OOD_CONDITIONS:
            condition_rows = [
                row for row in trajectory_results if row["condition"] == condition
            ]
            condition_counts[condition] = len(condition_rows)
            condition_sum_nrmse[condition] = {
                k: float(sum(row[f"nRMSE_{k}"] for row in condition_rows))
                for k in args.eval_k
            }
            condition_nrmse[condition] = {
                k: (
                    condition_sum_nrmse[condition][k] / len(condition_rows)
                    if condition_rows
                    else float("nan")
                )
                for k in args.eval_k
            }

    del loader
    del dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset_name": pde,
        "test_split": split_info,
        "num_test_trajectories": int(num_trajectories),
        "num_evaluated_trajectories": int(count),
        "trajectory_length": shape_info["trajectory_length"],
        "data_shape": shape_info["data_shape"],
        "channels": shape_info["channels"],
        "nRMSE_per_k": nrmse,
        "sum_nrmse_per_k": dict(sum_nrmse),
        "trajectory_results": trajectory_results,
        "condition_nRMSE_per_k": condition_nrmse,
        "condition_sum_nrmse_per_k": condition_sum_nrmse,
        "condition_counts": condition_counts,
        "elapsed_seconds": elapsed,
    }


def run_cache_mode(
    strategy: SingleStepSupervised,
    cache_mode_on: bool,
    args: argparse.Namespace,
    data_dir: Path,
    device: torch.device,
    output_dir: Path,
    base_metadata: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    cache_label = "on" if cache_mode_on else "off"
    strategy.use_ttt_state_cache_inference = cache_mode_on
    strategy.use_ttt_state_cache_train = False
    strategy.eval()

    started_iso = datetime.now().isoformat(timespec="seconds")
    started_perf = time.perf_counter()

    print()
    print("=" * 78)
    print(f"== Official eval (cache mode: {cache_label}) ==")
    print("=" * 78)
    print(f"model_source:      {base_metadata['model_source']}")
    print(f"checkpoint:        {base_metadata.get('checkpoint')}")
    print(f"device:            {device}")
    print(f"datasets ({len(args.datasets)}): {', '.join(args.datasets)}")
    print(f"rollout_steps:     {args.rollout_steps}")
    print(f"eval horizons k:   {args.eval_k}")
    print(f"test_unroll_steps: {args.test_unrolling_steps}")
    print(f"batch_size:        {args.batch_size}")
    print(f"num_workers:       {args.num_workers}")
    print(f"total params:      {_format_int(params['total'])} ({params['total_m']})")
    print("-" * 78)

    per_dataset: dict[str, dict[str, Any]] = {}
    for pde in args.datasets:
        print(f"[eval:{cache_label}] dataset={pde} ...", flush=True)
        result = evaluate_dataset(strategy, args, data_dir, pde, device)
        per_dataset[pde] = result
        metric_str = "  ".join(f"nRMSE_{k}={result['nRMSE_per_k'][k]:.6g}" for k in args.eval_k)
        print(
            f"  -> source={result['test_split']['source_file_name']} "
            f"sims={result['test_split']['selected_sim_ids']} "
            f"trajectories(test={result['num_test_trajectories']}, "
            f"evaluated={result['num_evaluated_trajectories']}) "
            f"T={result['trajectory_length']} shape={tuple(result['data_shape'])} "
            f"elapsed={result['elapsed_seconds']:.1f}s {metric_str}",
            flush=True,
        )

    macro: dict[int, float] = {}
    micro: dict[int, float] = {}
    for k in args.eval_k:
        vals = [
            per_dataset[pde]["nRMSE_per_k"][k]
            for pde in args.datasets
            if per_dataset[pde]["num_evaluated_trajectories"] > 0
        ]
        macro[k] = float(np.mean(vals)) if vals else float("nan")
        total_sum = sum(per_dataset[pde]["sum_nrmse_per_k"][k] for pde in args.datasets)
        total_count = sum(per_dataset[pde]["num_evaluated_trajectories"] for pde in args.datasets)
        micro[k] = total_sum / total_count if total_count > 0 else float("nan")

    condition_aggregate: dict[str, dict[str, Any]] = {}
    if args.id_ood_test:
        for condition in ID_OOD_CONDITIONS:
            condition_macro: dict[int, float] = {}
            condition_micro: dict[int, float] = {}
            condition_count = sum(
                per_dataset[pde]["condition_counts"][condition]
                for pde in args.datasets
            )
            for k in args.eval_k:
                pde_values = [
                    per_dataset[pde]["condition_nRMSE_per_k"][condition][k]
                    for pde in args.datasets
                    if per_dataset[pde]["condition_counts"][condition] > 0
                ]
                condition_macro[k] = (
                    float(np.mean(pde_values)) if pde_values else float("nan")
                )
                condition_sum = sum(
                    per_dataset[pde]["condition_sum_nrmse_per_k"][condition][k]
                    for pde in args.datasets
                )
                condition_micro[k] = (
                    condition_sum / condition_count
                    if condition_count > 0
                    else float("nan")
                )
            condition_aggregate[condition] = {
                "macro": condition_macro,
                "micro": condition_micro,
                "count": condition_count,
            }

    total_trajectories = sum(r["num_evaluated_trajectories"] for r in per_dataset.values())
    elapsed_total = time.perf_counter() - started_perf
    ended_iso = datetime.now().isoformat(timespec="seconds")

    print("-" * 78)
    print("[aggregate:%s] macro: %s" % (cache_label, "  ".join(f"nRMSE_{k}={macro[k]:.6g}" for k in args.eval_k)))
    print("[aggregate:%s] micro: %s" % (cache_label, "  ".join(f"nRMSE_{k}={micro[k]:.6g}" for k in args.eval_k)))
    for condition, condition_result in condition_aggregate.items():
        print(
            "[aggregate:%s:%s] macro: %s"
            % (
                cache_label,
                condition,
                "  ".join(
                    f"nRMSE_{k}={condition_result['macro'][k]:.6g}"
                    for k in args.eval_k
                ),
            )
        )
    print(f"elapsed_total:     {elapsed_total:.1f}s")
    print("=" * 78)

    metadata = dict(base_metadata)
    metadata.update(
        {
            "cache_mode": cache_label,
            "started_at": started_iso,
            "ended_at": ended_iso,
            "elapsed_seconds": elapsed_total,
        }
    )
    payload = {
        "metadata": metadata,
        "params": params,
        "per_dataset": {
            pde: {
                "dataset_name": r["dataset_name"],
                "test_split": r["test_split"],
                "num_test_trajectories": r["num_test_trajectories"],
                "num_evaluated_trajectories": r["num_evaluated_trajectories"],
                "trajectory_length": r["trajectory_length"],
                "data_shape": r["data_shape"],
                "channels": r["channels"],
                **{f"nRMSE_{k}": r["nRMSE_per_k"][k] for k in args.eval_k},
                "conditions": {
                    condition: {
                        "count": r["condition_counts"][condition],
                        **{
                            f"nRMSE_{k}": r["condition_nRMSE_per_k"][condition][k]
                            for k in args.eval_k
                        },
                    }
                    for condition in ID_OOD_CONDITIONS
                } if args.id_ood_test else {},
                "trajectories": r["trajectory_results"],
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for pde, r in per_dataset.items()
        },
        "aggregate": {
            "macro": {f"nRMSE_{k}": macro[k] for k in args.eval_k},
            "micro": {f"nRMSE_{k}": micro[k] for k in args.eval_k},
            "conditions": {
                condition: {
                    "count": result["count"],
                    "macro": {
                        f"nRMSE_{k}": result["macro"][k] for k in args.eval_k
                    },
                    "micro": {
                        f"nRMSE_{k}": result["micro"][k] for k in args.eval_k
                    },
                }
                for condition, result in condition_aggregate.items()
            },
            "total_evaluated_trajectories": int(total_trajectories),
        },
    }

    json_path = output_dir / f"results_cache_{cache_label}.json"
    csv_path = output_dir / f"results_cache_{cache_label}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "dataset",
        "source_dataset",
        "source_file",
        "source_num_simulations",
        "source_num_frames",
        "selected_sim_ids",
        "selected_num_simulations",
        "num_test_trajectories",
        "num_evaluated_trajectories",
        "trajectory_length",
        "shape",
        *[f"nRMSE_{k}" for k in args.eval_k],
        "elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pde in args.datasets:
            r = per_dataset[pde]
            row = {
                "dataset": pde,
                "source_dataset": r["test_split"]["source_dataset_name"],
                "source_file": r["test_split"]["source_file_name"],
                "source_num_simulations": r["test_split"]["source_num_simulations"],
                "source_num_frames": r["test_split"]["source_num_frames"],
                "selected_sim_ids": ",".join(
                    str(sim_id) for sim_id in r["test_split"]["selected_sim_ids"]
                ),
                "selected_num_simulations": r["test_split"]["selected_num_simulations"],
                "num_test_trajectories": r["num_test_trajectories"],
                "num_evaluated_trajectories": r["num_evaluated_trajectories"],
                "trajectory_length": r["trajectory_length"],
                "shape": "x".join(str(d) for d in r["data_shape"]),
                "elapsed_seconds": f"{r['elapsed_seconds']:.3f}",
            }
            for k in args.eval_k:
                row[f"nRMSE_{k}"] = f"{r['nRMSE_per_k'][k]:.6g}"
            writer.writerow(row)
        writer.writerow(
            {
                "dataset": "macro_avg",
                **{f"nRMSE_{k}": f"{macro[k]:.6g}" for k in args.eval_k},
            }
        )
        writer.writerow(
            {
                "dataset": "micro_avg",
                "num_evaluated_trajectories": total_trajectories,
                "elapsed_seconds": f"{elapsed_total:.3f}",
                **{f"nRMSE_{k}": f"{micro[k]:.6g}" for k in args.eval_k},
            }
        )

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    if args.id_ood_test:
        condition_csv_path = output_dir / f"results_conditions_cache_{cache_label}.csv"
        trajectory_csv_path = output_dir / f"results_trajectories_cache_{cache_label}.csv"
        condition_fieldnames = [
            "dataset",
            "condition",
            "count",
            *[f"nRMSE_{k}" for k in args.eval_k],
        ]
        with condition_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=condition_fieldnames)
            writer.writeheader()
            for pde in args.datasets:
                r = per_dataset[pde]
                for condition in ID_OOD_CONDITIONS:
                    writer.writerow(
                        {
                            "dataset": pde,
                            "condition": condition,
                            "count": r["condition_counts"][condition],
                            **{
                                f"nRMSE_{k}": (
                                    f"{r['condition_nRMSE_per_k'][condition][k]:.6g}"
                                )
                                for k in args.eval_k
                            },
                        }
                    )
            for condition, result in condition_aggregate.items():
                for aggregate_name in ("macro", "micro"):
                    writer.writerow(
                        {
                            "dataset": f"{aggregate_name}_avg",
                            "condition": condition,
                            "count": result["count"],
                            **{
                                f"nRMSE_{k}": f"{result[aggregate_name][k]:.6g}"
                                for k in args.eval_k
                            },
                        }
                    )

        trajectory_fieldnames = [
            "dataset",
            "sim_id",
            "condition",
            "seed",
            "parameter_overrides",
            "numerical_overrides",
            *[f"nRMSE_{k}" for k in args.eval_k],
        ]
        with trajectory_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=trajectory_fieldnames)
            writer.writeheader()
            for pde in args.datasets:
                for row in per_dataset[pde]["trajectory_results"]:
                    writer.writerow(
                        {
                            "dataset": pde,
                            **row,
                            "parameter_overrides": json.dumps(
                                row["parameter_overrides"], sort_keys=True
                            ),
                            "numerical_overrides": json.dumps(
                                row["numerical_overrides"], sort_keys=True
                            ),
                        }
                    )
        print(f"wrote {condition_csv_path}")
        print(f"wrote {trajectory_csv_path}")
    return payload


def select_cache_modes(args: argparse.Namespace, resolved_token_mixer: str, is_pretrained: bool) -> list[bool]:
    cache_capable = (not is_pretrained) and resolved_token_mixer == "ttt_sequence"
    if args.cache_mode == "auto":
        return [False, True] if cache_capable else [False]
    if not cache_capable and args.cache_mode != "off":
        print(
            f"[note] cache mode {args.cache_mode!r} is invalid for "
            f"{'pretrained' if is_pretrained else resolved_token_mixer}; forcing off."
        )
        return [False]
    if args.cache_mode == "off":
        return [False]
    if args.cache_mode == "on":
        return [True]
    return [False, True]


def main() -> None:
    args = parse_args()

    if args.id_ood_test:
        args.strict_test_split = True
    if args.strict_test_split:
        if len(args.datasets) != len(set(args.datasets)):
            raise SystemExit("Strict test evaluation does not allow duplicate dataset names.")
        if args.test_unrolling_steps != 29 or args.rollout_steps != 30:
            raise SystemExit(
                "Strict test evaluation requires test_unrolling_steps=29 "
                "and rollout_steps=30."
            )
        if args.max_batches_per_dataset is not None:
            raise SystemExit(
                "Strict test evaluation requires the complete split; "
                "remove --max-batches-per-dataset."
            )
    if args.id_ood_test:
        unsupported = sorted(set(args.datasets) - set(DATASET_NAMES))
        if unsupported:
            raise SystemExit(
                "--id-ood-test only supports the 17 APE2D PDE datasets; "
                f"unsupported datasets: {unsupported}."
            )

    if args.rollout_steps <= max(args.eval_k):
        raise SystemExit(
            f"--rollout-steps ({args.rollout_steps}) must be greater than max(--eval-k)={max(args.eval_k)}."
        )
    if args.rollout_steps > args.test_unrolling_steps + 1:
        raise SystemExit(
            f"--rollout-steps ({args.rollout_steps}) requires at least "
            f"test_unrolling_steps={args.rollout_steps - 1}; current value is "
            f"{args.test_unrolling_steps}."
        )

    work_dir = _expand(args.work_dir) or Path("~/working").expanduser().resolve()
    data_dir = _expand(args.data_dir) or (work_dir / "datasets")
    run_root = _expand(args.run_root) or (work_dir / "runs_v2")
    checkpoint_path = _expand(args.checkpoint_path)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    args.id_ood_manifest = (
        load_id_ood_manifest(
            data_dir,
            args.datasets,
            args.sample_size,
            args.downsample_factor,
        )
        if args.id_ood_test
        else None
    )
    data_dir_warning = (
        "official" not in str(data_dir).lower()
        and "ape2d_full" not in str(data_dir).lower()
        and not args.id_ood_test
    )
    if data_dir_warning:
        print(
            "[warn] data_dir does not look like the official test set. "
            "For official_data_eval_report.md, pass "
            "--data-dir ~/working/datasets_official."
        )

    is_pretrained = checkpoint_path is None
    resolved_token_mixer = "attention" if is_pretrained else _resolve_token_mixer(
        args.token_mixer_type,
        args.use_ttt_window_attention,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    default_name = "pretrained_%s" % ((args.subfolder or "root").replace("/", "_"))
    result_name = args.run_name if not is_pretrained else default_name
    output_dir = (
        _expand(args.output_dir)
        if args.output_dir is not None
        else (run_root / result_name / "test_results" / timestamp)
    )
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"config:            {args.config}")
    print(f"work_dir:          {work_dir}")
    print(f"data_dir:          {data_dir}")
    print(f"run_root:          {run_root}")
    print(f"run_name:          {args.run_name}")
    print(f"model_source:      {'from_pretrained' if is_pretrained else 'checkpoint'}")
    print(f"checkpoint_path:   {checkpoint_path}")
    print(f"hf_model_source:   {args.model_source}")
    print(f"subfolder:         {args.subfolder}")
    print(f"output_dir:        {output_dir}")
    print(f"model_type:        {args.model_type}")
    print(f"token_mixer_type:  {args.token_mixer_type} resolved={resolved_token_mixer}")
    print(f"sample_size:       {args.sample_size}")
    print(f"downsample_factor: {args.downsample_factor}")
    print(f"test_unroll_steps: {args.test_unrolling_steps}")
    print(f"dataset_profile:   {args.dataset_profile}")
    print(f"strict test split: {args.strict_test_split}")
    print(f"legacy all sims:   {args.legacy_all_source_sims}")
    print(f"ID/OOD test-only:  {args.id_ood_test}")
    print(f"torch:             {torch.__version__}")
    print(f"cuda available:    {torch.cuda.is_available()}")
    print(f"device:            {device}")

    strategy = (
        build_pretrained_strategy(args)
        if is_pretrained
        else build_checkpoint_strategy(args, checkpoint_path)
    )
    strategy = strategy.to(device)
    strategy.eval()

    total, trainable = count_parameters(strategy.model)
    params = {
        "total": int(total),
        "trainable": int(trainable),
        "total_m": _format_million(total),
        "trainable_m": _format_million(trainable),
    }

    base_metadata = {
        "model_source": "from_pretrained" if is_pretrained else "checkpoint",
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "hf_model_source": args.model_source,
        "subfolder": args.subfolder,
        "work_dir": str(work_dir),
        "data_dir": str(data_dir),
        "data_dir_warning_non_official": data_dir_warning,
        "dataset_profile": args.dataset_profile,
        "strict_test_split": args.strict_test_split,
        "legacy_all_source_sims": args.legacy_all_source_sims,
        "id_ood_test": args.id_ood_test,
        "id_ood_manifest_path": (
            str(data_dir / "manifest.json") if args.id_ood_test else None
        ),
        "run_root": str(run_root),
        "run_name": args.run_name,
        "model_type": args.model_type,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "patch_size": args.patch_size,
        "periodic": args.periodic,
        "carrier_token_active": args.carrier_token_active,
        "token_mixer_type": args.token_mixer_type,
        "resolved_token_mixer_type": resolved_token_mixer,
        "use_ttt_window_attention": args.use_ttt_window_attention,
        "use_ttt_state_cache_train": args.use_ttt_state_cache_train,
        "ttt_layer_type": args.ttt_layer_type,
        "ttt_mini_batch_size": args.ttt_mini_batch_size,
        "ttt_base_lr": args.ttt_base_lr,
        "ttt_use_gate": args.ttt_use_gate,
        "ttt_scan_checkpoint_group_size": args.ttt_scan_checkpoint_group_size,
        "vittt_inner_lr": args.vittt_inner_lr,
        "vittt_padding_mode": args.vittt_padding_mode,
        "attention_ttt_type": args.attention_ttt_type,
        "attention_ttt_gate_init": args.attention_ttt_gate_init,
        "attention_ttt_bidirectional": args.attention_ttt_bidirectional,
        "downsample_factor": args.downsample_factor,
        "sample_size": args.sample_size,
        "test_unrolling_steps": args.test_unrolling_steps,
        "max_channels": args.max_channels,
        "rollout_steps": args.rollout_steps,
        "eval_k": list(args.eval_k),
        "datasets": list(args.datasets),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "max_batches_per_dataset": args.max_batches_per_dataset,
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "config_path": str(args.config) if args.config else None,
        "output_dir": str(output_dir),
    }

    cache_modes = select_cache_modes(args, resolved_token_mixer, is_pretrained)
    payloads: dict[str, dict[str, Any]] = {}
    overall_start = datetime.now().isoformat(timespec="seconds")
    overall_t0 = time.perf_counter()
    for cache_mode_on in cache_modes:
        label = "on" if cache_mode_on else "off"
        payloads[label] = run_cache_mode(
            strategy=strategy,
            cache_mode_on=cache_mode_on,
            args=args,
            data_dir=data_dir,
            device=device,
            output_dir=output_dir,
            base_metadata=base_metadata,
            params=params,
        )
    overall_elapsed = time.perf_counter() - overall_t0
    overall_end = datetime.now().isoformat(timespec="seconds")

    summary = {
        "metadata": {
            **base_metadata,
            "cache_modes_run": list(payloads.keys()),
            "overall_started_at": overall_start,
            "overall_ended_at": overall_end,
            "overall_elapsed_seconds": overall_elapsed,
        },
        "params": params,
        "aggregates_by_cache_mode": {
            label: payload["aggregate"] for label, payload in payloads.items()
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")
    print(f"overall_elapsed: {overall_elapsed:.1f}s")


if __name__ == "__main__":
    main()
