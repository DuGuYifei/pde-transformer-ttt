"""Paper-style nRMSE evaluation for trained TTT PDE-Transformer checkpoints.

Loads a checkpoint and iterates the full test loader of every PDE dataset
separately (one MultiDataModule per PDE), then reports:
    * nRMSE_1, nRMSE_10, nRMSE_20 per dataset
    * macro_avg (mean over datasets) and micro_avg (mean over all trajectories)
    * total / trainable parameters
    * start / end timestamps and elapsed seconds
Runs once per requested TTT-cache-inference mode (off / on / both).
Writes a timestamped directory with one JSON and one CSV per cache mode.

Run from the server_example directory (same pattern as the train script):

    CUDA_VISIBLE_DEVICES=0 python test_ttt_ape_xxl_server.py \\
        --config train_ttt_ape_xxl_server.yaml \\
        --cache-mode both
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_ttt_ape_xxl_server import (
    DEFAULT_CONFIG,
    FULL_DATASET_NAMES,
    _load_config,
    _path_or_none,
    apply_sims_split_override,
    build_data_module,
    build_strategy,
)


DEFAULT_EVAL_K = (1, 10, 20)
DEFAULT_ROLLOUT_STEPS = 29


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, remaining = config_parser.parse_known_args()
    config = _load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Run nRMSE evaluation on a trained TTT PDE-Transformer checkpoint.",
        parents=[config_parser],
    )
    parser.set_defaults(config=config_args.config)

    parser.add_argument("--work-dir", type=Path, default=_path_or_none(config["work_dir"]))
    parser.add_argument("--data-dir", type=Path, default=_path_or_none(config["data_dir"]))
    parser.add_argument("--run-root", type=Path, default=_path_or_none(config["run_root"]))
    parser.add_argument("--run-name", type=str, default=config["run_name"])
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("PDE-S", "PDE-B", "PDE-L"),
        default=config.get("model_type", "PDE-S"),
        help=(
            "Model size. Picked up from yaml's model_type if present (newer "
            "train script), else falls back to PDE-S (committed train script "
            "hardcodes PDE-S inside build_strategy)."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help=(
            "Path to the .ckpt file to evaluate. REQUIRED — auto-picking the latest "
            "last*.ckpt is intentionally disabled for test runs (it is not "
            "necessarily the best-val checkpoint). Inspect "
            "<run_root>/<run_name>/checkpoints/ and choose explicitly."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--num-workers", type=int, default=config["num_workers"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--downsample-factor", type=int, default=config.get("downsample_factor", 2))
    parser.add_argument("--sample-size", type=int, default=config.get("sample_size", 128))
    parser.add_argument(
        "--use-ttt-window-attention",
        action=argparse.BooleanOptionalAction,
        default=config.get("use_ttt_window_attention", True),
        help="Must match how the checkpoint was trained (plain baseline vs TTT).",
    )
    parser.set_defaults(
        sims_split_joint_train=config.get("sims_split_joint_train"),
        sims_split_joint_test=config.get("sims_split_joint_test"),
        sims_split_gs_train=config.get("sims_split_gs_train"),
        sims_split_gs_test=config.get("sims_split_gs_test"),
    )

    parser.add_argument(
        "--cache-mode",
        type=str,
        choices=("off", "on", "both"),
        default="both",
        help="TTT state cache mode during inference. 'both' runs off then on.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
        help="num_frames passed to strategy.predict; must exceed max(--eval-k).",
    )
    parser.add_argument(
        "--eval-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVAL_K),
        help="Rollout horizons (frame indices) at which to report nRMSE.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(FULL_DATASET_NAMES),
        help="Subset of PDE dataset names to evaluate (default: all).",
    )
    parser.add_argument(
        "--max-batches-per-dataset",
        type=int,
        default=None,
        help="Optional cap on number of batches per dataset (debugging).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)

    return parser.parse_args(remaining)


def _format_million(n: int) -> str:
    return f"{n / 1e6:.2f}M"


def _format_int(n: int) -> str:
    return f"{n:,}"


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def per_trajectory_nrmse(pred_k: np.ndarray, ref_k: np.ndarray) -> np.ndarray:
    """Per-sample nRMSE at one rollout step. Inputs shape: (B, C, H, W).

    Returns shape (B,) with sqrt( mean_spatial((pred-ref)^2) / mean_spatial(ref^2) ).
    """
    axes = tuple(range(1, pred_k.ndim))
    mse_pred = np.mean((pred_k - ref_k) ** 2, axis=axes)
    mse_zero = np.mean(ref_k ** 2, axis=axes)
    return np.sqrt(mse_pred / np.clip(mse_zero, 1e-30, None))


def inspect_first_batch(loader) -> dict[str, Any]:
    """Pull one batch to recover shape / channel info. Loader is recreated by caller."""
    batch = next(iter(loader))
    data = batch["data"] if isinstance(batch, dict) else batch[0]
    # data shape: (B, T, C, H, W)
    if data.ndim != 5:
        raise RuntimeError(f"Unexpected data shape {tuple(data.shape)}; expected (B, T, C, H, W).")
    _, t, c, h, w = data.shape
    return {"trajectory_length": int(t), "data_shape": [int(c), int(h), int(w)], "channels": int(c)}


def evaluate_dataset(
    strategy,
    data_dir: Path,
    pde: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    rollout_steps: int,
    eval_k: list[int],
    max_batches: int | None,
    downsample_factor: int = 2,
) -> dict[str, Any]:
    dm = build_data_module(data_dir, [pde], batch_size, num_workers, downsample_factor=downsample_factor)
    dm.setup(stage="test")
    loader = dm.test_dataloader()
    num_trajectories = len(dm.set_test) if dm.set_test is not None else 0

    # Use a fresh iterator for shape inspection then a second iterator for the eval pass.
    info_loader = dm.test_dataloader()
    shape_info = inspect_first_batch(info_loader)
    del info_loader

    sum_nrmse = {k: 0.0 for k in eval_k}
    count = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            prediction, reference = strategy.predict(
                batch, device=device, num_frames=rollout_steps
            )
            prediction = np.asarray(prediction)
            reference = np.asarray(reference)
            # prediction / reference shape: (B, num_frames, C, H, W); frame 0 == input.

            batch_b = prediction.shape[0]
            for k in eval_k:
                per_traj = per_trajectory_nrmse(prediction[:, k], reference[:, k])
                sum_nrmse[k] += float(per_traj.sum())
            count += batch_b

    elapsed = time.perf_counter() - t0

    if count == 0:
        nrmse = {k: float("nan") for k in eval_k}
    else:
        nrmse = {k: sum_nrmse[k] / count for k in eval_k}

    # Free the (test-only) datamodule before moving on.
    del loader
    del dm
    gc.collect()

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
    strategy,
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

    # ---- Header --------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"== Test plan (cache mode: {cache_label}) ==")
    print("=" * 78)
    print(f"checkpoint:        {base_metadata['checkpoint']}")
    print(f"device:            {device}")
    print(f"datasets ({len(args.datasets)}): {', '.join(args.datasets)}")
    print(f"rollout_steps:     {args.rollout_steps}")
    print(f"eval horizons k:   {args.eval_k}")
    print(f"batch_size:        {args.batch_size}")
    print(f"num_workers:       {args.num_workers}")
    print(
        f"total params:      {_format_int(params['total'])}  ({params['total_m']})"
    )
    print(
        f"trainable params:  {_format_int(params['trainable'])}  ({params['trainable_m']})"
    )
    print(f"started_at:        {started_iso}")
    print("-" * 78)

    per_dataset: dict[str, dict[str, Any]] = {}

    for pde in args.datasets:
        print(f"[eval:{cache_label}] dataset={pde} ...", flush=True)
        result = evaluate_dataset(
            strategy=strategy,
            data_dir=data_dir,
            pde=pde,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            rollout_steps=args.rollout_steps,
            eval_k=args.eval_k,
            max_batches=args.max_batches_per_dataset,
            downsample_factor=args.downsample_factor,
        )
        per_dataset[pde] = result
        metric_str = "  ".join(
            f"nRMSE_{k}={result['nRMSE_per_k'][k]:.6g}" for k in args.eval_k
        )
        print(
            f"  -> trajectories(test={result['num_test_trajectories']}, "
            f"evaluated={result['num_evaluated_trajectories']})  "
            f"T={result['trajectory_length']}  "
            f"shape={tuple(result['data_shape'])}  "
            f"channels={result['channels']}  "
            f"elapsed={result['elapsed_seconds']:.1f}s  "
            f"{metric_str}",
            flush=True,
        )

    # ---- Aggregates ----------------------------------------------------------
    macro = {}
    micro = {}
    for k in args.eval_k:
        per_ds_vals = [
            per_dataset[pde]["nRMSE_per_k"][k]
            for pde in args.datasets
            if per_dataset[pde]["num_evaluated_trajectories"] > 0
        ]
        macro[k] = float(np.mean(per_ds_vals)) if per_ds_vals else float("nan")

        total_sum = sum(per_dataset[pde]["sum_nrmse_per_k"][k] for pde in args.datasets)
        total_count = sum(
            per_dataset[pde]["num_evaluated_trajectories"] for pde in args.datasets
        )
        micro[k] = (total_sum / total_count) if total_count > 0 else float("nan")

    total_trajectories = sum(
        per_dataset[pde]["num_evaluated_trajectories"] for pde in args.datasets
    )

    ended_iso = datetime.now().isoformat(timespec="seconds")
    elapsed_total = time.perf_counter() - started_perf

    print("-" * 78)
    print(
        f"[aggregate:{cache_label}] macro: "
        + "  ".join(f"nRMSE_{k}={macro[k]:.6g}" for k in args.eval_k)
    )
    print(
        f"[aggregate:{cache_label}] micro: "
        + "  ".join(f"nRMSE_{k}={micro[k]:.6g}" for k in args.eval_k)
    )
    print(f"ended_at:          {ended_iso}")
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
                "num_test_trajectories": "",
                "num_evaluated_trajectories": "",
                "trajectory_length": "",
                "shape": "",
                "elapsed_seconds": "",
                **{f"nRMSE_{k}": f"{macro[k]:.6g}" for k in args.eval_k},
            }
        )
        writer.writerow(
            {
                "dataset": "micro_avg",
                "num_test_trajectories": "",
                "num_evaluated_trajectories": total_trajectories,
                "trajectory_length": "",
                "shape": "",
                "elapsed_seconds": f"{elapsed_total:.3f}",
                **{f"nRMSE_{k}": f"{micro[k]:.6g}" for k in args.eval_k},
            }
        )

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return payload


def main() -> None:
    args = parse_args()

    if args.rollout_steps <= max(args.eval_k):
        raise SystemExit(
            f"--rollout-steps ({args.rollout_steps}) must be strictly greater than "
            f"max(--eval-k)={max(args.eval_k)}; frame index k must exist in the rollout."
        )

    work_dir = args.work_dir.expanduser().resolve()
    data_dir = (args.data_dir or work_dir / "datasets").expanduser().resolve()
    run_root = (args.run_root or work_dir / "ttt_cache_experiments").expanduser().resolve()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_root / args.run_name / "test_results" / timestamp)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"config:            {args.config}")
    print(f"work_dir:          {work_dir}")
    print(f"data_dir:          {data_dir}")
    print(f"run_root:          {run_root}")
    print(f"run_name:          {args.run_name}")
    print(f"model_type:        {args.model_type}")
    print(f"checkpoint_path:   {checkpoint_path}")
    print(f"output_dir:        {output_dir}")
    print(f"torch:             {torch.__version__}")
    print(f"cuda available:    {torch.cuda.is_available()}")
    print(
        f"cuda device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
    )
    print(f"device:            {device}")

    # Build strategy and load checkpoint weights.
    # Forward model_type only if the imported build_strategy accepts it
    # (newer train script). On the committed server version build_strategy
    # hardcodes PDE-S and does not accept model_type — we skip the kwarg
    # there and warn if a non-default size was requested.
    build_kwargs: dict[str, Any] = {
        "use_ttt_state_cache_inference": False,  # toggled per cache mode below
        "use_ttt_state_cache_train": False,
    }
    build_params = inspect.signature(build_strategy).parameters
    if "model_type" in build_params:
        build_kwargs["model_type"] = args.model_type
    elif args.model_type != "PDE-S":
        print(
            f"[warn] --model-type={args.model_type} requested, but the imported "
            "build_strategy hardcodes PDE-S. Loading will likely fail on "
            "non-PDE-S checkpoints; update the train script to support "
            "model_type or rerun with --model-type PDE-S."
        )
    if "sample_size" in build_params:
        build_kwargs["sample_size"] = args.sample_size
    if "use_ttt_window_attention" in build_params:
        build_kwargs["use_ttt_window_attention"] = args.use_ttt_window_attention

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

    strategy = build_strategy(args.seed, **build_kwargs)

    print(f"loading state_dict from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise SystemExit(
            f"Checkpoint at {checkpoint_path} does not contain 'state_dict' "
            "(unexpected format)."
        )
    missing, unexpected = strategy.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    strategy = strategy.to(device)

    total, trainable = count_parameters(strategy.model)
    params = {
        "total": int(total),
        "trainable": int(trainable),
        "total_m": _format_million(total),
        "trainable_m": _format_million(trainable),
    }

    base_metadata = {
        "checkpoint": str(checkpoint_path),
        "work_dir": str(work_dir),
        "data_dir": str(data_dir),
        "run_root": str(run_root),
        "run_name": args.run_name,
        "model_type": args.model_type,
        "use_ttt_window_attention": args.use_ttt_window_attention,
        "downsample_factor": args.downsample_factor,
        "rollout_steps": args.rollout_steps,
        "eval_k": list(args.eval_k),
        "datasets": list(args.datasets),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "max_batches_per_dataset": args.max_batches_per_dataset,
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_device_count": (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        ),
        "config_path": str(args.config) if args.config else None,
        "output_dir": str(output_dir),
    }

    if not args.use_ttt_window_attention and args.cache_mode != "off":
        print(
            "[note] plain baseline has no TTT state; forcing --cache-mode off "
            f"(was {args.cache_mode!r})."
        )
        args.cache_mode = "off"

    if args.cache_mode == "off":
        cache_modes = [False]
    elif args.cache_mode == "on":
        cache_modes = [True]
    else:
        cache_modes = [False, True]

    overall_start_iso = datetime.now().isoformat(timespec="seconds")
    overall_start_perf = time.perf_counter()
    payloads = {}
    for cache_mode_on in cache_modes:
        payloads["on" if cache_mode_on else "off"] = run_cache_mode(
            strategy=strategy,
            cache_mode_on=cache_mode_on,
            args=args,
            data_dir=data_dir,
            device=device,
            output_dir=output_dir,
            base_metadata=base_metadata,
            params=params,
        )
    overall_elapsed = time.perf_counter() - overall_start_perf
    overall_end_iso = datetime.now().isoformat(timespec="seconds")

    summary = {
        "metadata": {
            **base_metadata,
            "cache_modes_run": [
                "on" if m else "off" for m in cache_modes
            ],
            "overall_started_at": overall_start_iso,
            "overall_ended_at": overall_end_iso,
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

    print()
    print(f"overall_started_at: {overall_start_iso}")
    print(f"overall_ended_at:   {overall_end_iso}")
    print(f"overall_elapsed:    {overall_elapsed:.1f}s")


if __name__ == "__main__":
    main()
