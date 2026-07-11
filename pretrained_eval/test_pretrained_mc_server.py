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
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pdetransformer.core.mixed_channels import PDETransformer, SingleStepSupervised
from pdetransformer.data import MultiDataModule


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
    "vittt_padding_mode": "zero",
    "attention_ttt_type": "ttt_sequence",
    "attention_ttt_gate_init": 0.1,
    "attention_ttt_bidirectional": True,
    "global_ttt_stage_names": [],
    "global_ttt_inner_lr": 1.0,
    "global_ttt_gate_init": 0.0,
    "global_ttt_key_norm": True,
    "temporal_ttt_enabled": False,
    "temporal_ttt_layer_type": "mlp",
    "temporal_ttt_mini_batch_size": 64,
    "temporal_ttt_base_lr": 1.0,
    "temporal_ttt_gate_init": 0.1,
    "temporal_ttt_use_output_gate": False,
    "temporal_ttt_scan_checkpoint_group_size": 0,
    "batch_size": 8,
    "num_workers": 2,
    "seed": 42,
    "downsample_factor": 2,
    "sample_size": 128,
    "test_unrolling_steps": 29,
    "max_channels": 2,
    "checkpoint_path": None,
}

DEFAULT_EVAL_K = (1, 10, 20, 29)
DEFAULT_ROLLOUT_STEPS = 30


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
        choices=("attention", "ttt_sequence", "vittt", "attention_ttt"),
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
    parser.add_argument(
        "--temporal-ttt-enabled",
        action=argparse.BooleanOptionalAction,
        default=cfg["temporal_ttt_enabled"],
    )
    parser.add_argument(
        "--temporal-ttt-layer-type",
        choices=("linear", "mlp"),
        default=cfg["temporal_ttt_layer_type"],
    )
    parser.add_argument(
        "--temporal-ttt-mini-batch-size",
        type=int,
        default=cfg["temporal_ttt_mini_batch_size"],
    )
    parser.add_argument("--temporal-ttt-base-lr", type=float, default=cfg["temporal_ttt_base_lr"])
    parser.add_argument("--temporal-ttt-gate-init", type=float, default=cfg["temporal_ttt_gate_init"])
    parser.add_argument(
        "--temporal-ttt-use-output-gate",
        action=argparse.BooleanOptionalAction,
        default=cfg["temporal_ttt_use_output_gate"],
    )
    parser.add_argument(
        "--temporal-ttt-scan-checkpoint-group-size",
        type=int,
        default=cfg["temporal_ttt_scan_checkpoint_group_size"],
    )

    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--downsample-factor", type=int, default=cfg["downsample_factor"])
    parser.add_argument("--sample-size", type=int, default=cfg["sample_size"])
    parser.add_argument("--test-unrolling-steps", type=int, default=cfg["test_unrolling_steps"])
    parser.add_argument("--max-channels", type=int, default=cfg["max_channels"])

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
    parser.add_argument("--datasets", type=str, nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--max-batches-per-dataset", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-nonstrict-load",
        action="store_true",
        help="Do not fail on missing/unexpected checkpoint keys.",
    )

    parser.set_defaults(global_ttt_stage_names=cfg["global_ttt_stage_names"])
    return parser.parse_args(remaining)


def build_data_module(
    data_dir: Path,
    dataset_names: list[str],
    batch_size: int,
    num_workers: int,
    downsample_factor: int,
    test_unrolling_steps: int,
    max_channels: int,
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
    }
    return MultiDataModule(**params_data)


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
        temporal_ttt_enabled=args.temporal_ttt_enabled,
        temporal_ttt_layer_type=args.temporal_ttt_layer_type,
        temporal_ttt_mini_batch_size=args.temporal_ttt_mini_batch_size,
        temporal_ttt_base_lr=args.temporal_ttt_base_lr,
        temporal_ttt_gate_init=args.temporal_ttt_gate_init,
        temporal_ttt_use_output_gate=args.temporal_ttt_use_output_gate,
        temporal_ttt_scan_checkpoint_group_size=args.temporal_ttt_scan_checkpoint_group_size,
    )
    strategy = SingleStepSupervised(
        model=model,
        image_key=0,
        optimizer="adamw",
        use_ttt_state_cache_inference=False,
        use_ttt_state_cache_train=False,
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
        use_ttt_state_cache_inference=False,
        use_ttt_state_cache_train=False,
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
    dm = build_data_module(
        data_dir=data_dir,
        dataset_names=[pde],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        downsample_factor=args.downsample_factor,
        test_unrolling_steps=args.test_unrolling_steps,
        max_channels=args.max_channels,
    )
    dm.setup(stage="test")
    loader = dm.test_dataloader()
    num_trajectories = len(dm.set_test) if dm.set_test is not None else 0

    info_loader = dm.test_dataloader()
    shape_info = inspect_first_batch(info_loader)
    del info_loader

    sum_nrmse = {k: 0.0 for k in args.eval_k}
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
            count += batch_b

    elapsed = time.perf_counter() - t0
    nrmse = {
        k: (sum_nrmse[k] / count if count > 0 else float("nan"))
        for k in args.eval_k
    }

    del loader
    del dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset_name": pde,
        "num_test_trajectories": int(num_trajectories),
        "num_evaluated_trajectories": int(count),
        "trajectory_length": shape_info["trajectory_length"],
        "data_shape": shape_info["data_shape"],
        "channels": shape_info["channels"],
        "nRMSE_per_k": nrmse,
        "sum_nrmse_per_k": dict(sum_nrmse),
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
            f"  -> trajectories(test={result['num_test_trajectories']}, "
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

    total_trajectories = sum(r["num_evaluated_trajectories"] for r in per_dataset.values())
    elapsed_total = time.perf_counter() - started_perf
    ended_iso = datetime.now().isoformat(timespec="seconds")

    print("-" * 78)
    print("[aggregate:%s] macro: %s" % (cache_label, "  ".join(f"nRMSE_{k}={macro[k]:.6g}" for k in args.eval_k)))
    print("[aggregate:%s] micro: %s" % (cache_label, "  ".join(f"nRMSE_{k}={micro[k]:.6g}" for k in args.eval_k)))
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
                "num_test_trajectories": r["num_test_trajectories"],
                "num_evaluated_trajectories": r["num_evaluated_trajectories"],
                "trajectory_length": r["trajectory_length"],
                "data_shape": r["data_shape"],
                "channels": r["channels"],
                **{f"nRMSE_{k}": r["nRMSE_per_k"][k] for k in args.eval_k},
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for pde, r in per_dataset.items()
        },
        "aggregate": {
            "macro": {f"nRMSE_{k}": macro[k] for k in args.eval_k},
            "micro": {f"nRMSE_{k}": micro[k] for k in args.eval_k},
            "total_evaluated_trajectories": int(total_trajectories),
        },
    }

    json_path = output_dir / f"results_cache_{cache_label}.json"
    csv_path = output_dir / f"results_cache_{cache_label}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "dataset",
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
    return payload


def select_cache_modes(args: argparse.Namespace, resolved_token_mixer: str, is_pretrained: bool) -> list[bool]:
    cache_capable = (not is_pretrained) and (
        resolved_token_mixer == "ttt_sequence" or args.temporal_ttt_enabled
    )
    if args.cache_mode == "auto":
        if args.temporal_ttt_enabled:
            return [True]
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
    data_dir_warning = "official" not in str(data_dir).lower()
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
    print(f"temporal_ttt:      {args.temporal_ttt_enabled}")
    print(f"sample_size:       {args.sample_size}")
    print(f"downsample_factor: {args.downsample_factor}")
    print(f"test_unroll_steps: {args.test_unrolling_steps}")
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
        "temporal_ttt_enabled": args.temporal_ttt_enabled,
        "temporal_ttt_layer_type": args.temporal_ttt_layer_type,
        "temporal_ttt_mini_batch_size": args.temporal_ttt_mini_batch_size,
        "temporal_ttt_base_lr": args.temporal_ttt_base_lr,
        "temporal_ttt_gate_init": args.temporal_ttt_gate_init,
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
