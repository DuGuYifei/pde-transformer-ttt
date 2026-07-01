"""Evaluate an off-the-shelf pretrained PDE-Transformer on the server dataset.

Self-contained: it imports ONLY the installed ``pdetransformer`` package, so you
can drop this single file into any run directory (e.g. ``~/test/``) and run it --
no sibling ``server_example/`` folder required.

It reproduces the same nRMSE-per-PDE protocol and the same data pipeline
(``downsample_factor=2``, ``max_channels=2``, mean-std norm) as your own model's
test, then writes the same JSON/CSV schema -- so you can diff the ``macro_avg`` /
``micro_avg`` rows directly against your TTT model's ``results_cache_off.csv``.

The only thing that changes vs. your test: instead of loading one of YOUR
``.ckpt`` files it loads a published diffusers checkpoint via
``PDETransformer.from_pretrained`` -- by default the official mixed-channel small
model ``thuerey-group/pde-transformer / mc-s``.

Why this is a valid comparison:
    The official mc-s model is the SAME architecture your server model uses
    (in=2, out=2, sample_size=128, PDE-S, patch_size=4, periodic, no carrier
    token) but WITHOUT the TTT window attention. ``from_pretrained`` reads the
    published ``config.json``, builds the plain (no-TTT) net, and loads the
    weights cleanly. Because there is no TTT state, only cache-OFF inference is
    meaningful, so this script runs cache OFF only.

Data note (read before comparing numbers):
    The official model was trained at full resolution (downsample_factor=1) on
    standard-resolution APEBench; your server data is low-res + downsampled by 2.
    We keep your pipeline, so this is a zero-shot / out-of-distribution test of
    the off-the-shelf model on your data -- not comparable to the paper's
    headline MC-S numbers.

Typical run (from ~/test on the server, venv activated):

    CUDA_VISIBLE_DEVICES=0 python test_pretrained_mc_server.py \\
        --model-source ~/working/pretrained_weights/pde-transformer --subfolder mc-s

(or omit --model-source to auto-download from HuggingFace if the node has internet)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pdetransformer.core.mixed_channels import PDETransformer, SingleStepSupervised
from pdetransformer.data import MultiDataModule


# The 17 APEBench-xxl PDEs, same list/order as the server training/eval setup.
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

DEFAULT_EVAL_K = (1, 10, 20)
DEFAULT_ROLLOUT_STEPS = 29

# Defaults derived from the server low-res pdes_* training configs.
CONFIG_DEFAULTS: dict[str, Any] = {
    "work_dir": "~/working",
    "data_dir": None,
    "run_root": "~/working/ttt_cache_low_res",
    "batch_size": 8,
    "num_workers": 2,
    "seed": 42,
}


def _path_or_none(value: Any) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    return Path(value)


def _load_config_overrides(path: Path | None) -> dict[str, Any]:
    """Optionally read a server-style yaml and pull only the keys we care about.

    Tolerant of all the extra training keys in that yaml (unlike the strict
    server _load_config) so the same file can be passed here unchanged.
    """
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
    for key in CONFIG_DEFAULTS:
        if key in loaded:
            cfg[key] = loaded[key]
    return cfg


# ---------------------------------------------------------------------------
# Data pipeline -- IDENTICAL params to server_example.build_data_module so the
# pretrained model sees byte-for-byte the same inputs as your own test.
# ---------------------------------------------------------------------------
def build_data_module(
    data_dir: Path, dataset_names: list[str], batch_size: int, num_workers: int
) -> MultiDataModule:
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


# ---------------------------------------------------------------------------
# Metrics / helpers -- same nRMSE protocol as the server model evaluation.
# ---------------------------------------------------------------------------
def _format_million(n: int) -> str:
    return f"{n / 1e6:.2f}M"


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def per_trajectory_nrmse(pred_k: np.ndarray, ref_k: np.ndarray) -> np.ndarray:
    """Per-sample nRMSE at one rollout step. Inputs shape: (B, C, H, W)."""
    axes = tuple(range(1, pred_k.ndim))
    mse_pred = np.mean((pred_k - ref_k) ** 2, axis=axes)
    mse_zero = np.mean(ref_k ** 2, axis=axes)
    return np.sqrt(mse_pred / np.clip(mse_zero, 1e-30, None))


def inspect_first_batch(loader) -> dict[str, Any]:
    batch = next(iter(loader))
    data = batch["data"] if isinstance(batch, dict) else batch[0]
    if data.ndim != 5:
        raise RuntimeError(
            f"Unexpected data shape {tuple(data.shape)}; expected (B, T, C, H, W)."
        )
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
) -> dict[str, Any]:
    dm = build_data_module(data_dir, [pde], batch_size, num_workers)
    dm.setup(stage="test")
    loader = dm.test_dataloader()
    num_trajectories = len(dm.set_test) if dm.set_test is not None else 0

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
            # shape: (B, num_frames, C, H, W); frame 0 == input.

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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_pretrained_strategy(
    model_source: str,
    subfolder: str | None,
    seed: int,
) -> SingleStepSupervised:
    """Load a published diffusers PDETransformer and wrap it for evaluation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    load_kwargs: dict[str, Any] = {}
    if subfolder:
        load_kwargs["subfolder"] = subfolder

    print(f"loading pretrained model: source={model_source!r} subfolder={subfolder!r}")
    model = PDETransformer.from_pretrained(model_source, **load_kwargs)

    strategy = SingleStepSupervised(
        model=model,
        image_key=0,
        optimizer="adamw",
        use_ttt_state_cache_inference=False,  # published model has no TTT state
        use_ttt_state_cache_train=False,
    )
    strategy.learning_rate = 0.0  # unused during eval; attribute kept for safety
    return strategy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional server-style yaml; only work_dir/data_dir/run_root/"
        "batch_size/num_workers/seed are read from it.",
    )
    config_args, remaining = config_parser.parse_known_args()
    cfg = _load_config_overrides(config_args.config)

    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained mixed-channel PDE-Transformer on the server dataset.",
        parents=[config_parser],
    )
    parser.set_defaults(config=config_args.config)

    parser.add_argument("--work-dir", type=Path, default=_path_or_none(cfg["work_dir"]))
    parser.add_argument("--data-dir", type=Path, default=_path_or_none(cfg["data_dir"]))
    parser.add_argument("--run-root", type=Path, default=_path_or_none(cfg["run_root"]))
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])

    parser.add_argument(
        "--model-source",
        type=str,
        default="thuerey-group/pde-transformer",
        help="HuggingFace repo id or a local directory holding the <subfolder>/ "
        "with config.json + diffusion_pytorch_model.safetensors.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="mc-s",
        help="Subfolder inside --model-source (mc-s, mc-b, mc-l, ...). Empty string "
        "if --model-source already points at the leaf folder.",
    )

    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--eval-k", type=int, nargs="+", default=list(DEFAULT_EVAL_K))
    parser.add_argument("--datasets", type=str, nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--max-batches-per-dataset", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    return parser.parse_args(remaining)


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    per_dataset: dict[str, dict[str, Any]],
    params: dict[str, Any],
    base_metadata: dict[str, Any],
    started_iso: str,
    ended_iso: str,
    elapsed_total: float,
) -> dict[str, Any]:
    macro, micro = {}, {}
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

    print("-" * 78)
    print("[aggregate:off] macro: " + "  ".join(f"nRMSE_{k}={macro[k]:.6g}" for k in args.eval_k))
    print("[aggregate:off] micro: " + "  ".join(f"nRMSE_{k}={micro[k]:.6g}" for k in args.eval_k))
    print("=" * 78)

    metadata = dict(base_metadata)
    metadata.update(
        {
            "cache_mode": "off",
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

    json_path = output_dir / "results_cache_off.json"
    csv_path = output_dir / "results_cache_off.csv"

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


def main() -> None:
    args = parse_args()

    if args.rollout_steps <= max(args.eval_k):
        raise SystemExit(
            f"--rollout-steps ({args.rollout_steps}) must be strictly greater than "
            f"max(--eval-k)={max(args.eval_k)}; frame index k must exist in the rollout."
        )

    work_dir = args.work_dir.expanduser().resolve()
    data_dir = (args.data_dir or work_dir / "datasets").expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_subfolder = (args.subfolder or "root").replace("/", "_")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_root / "pretrained_eval" / safe_subfolder / timestamp)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"config:            {args.config}")
    print(f"work_dir:          {work_dir}")
    print(f"data_dir:          {data_dir}")
    print(f"run_root:          {run_root}")
    print(f"model_source:      {args.model_source}")
    print(f"subfolder:         {args.subfolder}")
    print(f"output_dir:        {output_dir}")
    print(f"torch:             {torch.__version__}")
    print(f"cuda available:    {torch.cuda.is_available()}")
    print(
        f"cuda device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
    )
    print(f"device:            {device}")

    strategy = build_pretrained_strategy(args.model_source, args.subfolder, args.seed)
    strategy = strategy.to(device)
    strategy.use_ttt_state_cache_inference = False
    strategy.use_ttt_state_cache_train = False
    strategy.eval()

    total, trainable = count_parameters(strategy.model)
    params = {
        "total": int(total),
        "trainable": int(trainable),
        "total_m": _format_million(total),
        "trainable_m": _format_million(trainable),
    }
    print(f"total params:      {total:,} ({params['total_m']})")

    model_id = f"{args.model_source}::{args.subfolder}" if args.subfolder else args.model_source
    base_metadata = {
        "checkpoint": model_id,
        "model_source": args.model_source,
        "subfolder": args.subfolder,
        "work_dir": str(work_dir),
        "data_dir": str(data_dir),
        "run_root": str(run_root),
        "model_type": "PDE-S (mixed channels, no TTT / from_pretrained)",
        "rollout_steps": args.rollout_steps,
        "eval_k": list(args.eval_k),
        "datasets": list(args.datasets),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "max_batches_per_dataset": args.max_batches_per_dataset,
        "torch_version": torch.__version__,
        "device": str(device),
        "config_path": str(args.config) if args.config else None,
        "output_dir": str(output_dir),
    }

    print()
    print("=" * 78)
    print("== Pretrained-model test plan (cache mode: off) ==")
    print("=" * 78)
    print(f"model:             {model_id}")
    print(f"datasets ({len(args.datasets)}): {', '.join(args.datasets)}")
    print(f"rollout_steps:     {args.rollout_steps}   eval horizons k: {args.eval_k}")
    print("-" * 78)

    started_iso = datetime.now().isoformat(timespec="seconds")
    started_perf = time.perf_counter()

    per_dataset: dict[str, dict[str, Any]] = {}
    for pde in args.datasets:
        print(f"[eval:off] dataset={pde} ...", flush=True)
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
            f"elapsed={result['elapsed_seconds']:.1f}s  {metric_str}",
            flush=True,
        )

    elapsed_total = time.perf_counter() - started_perf
    ended_iso = datetime.now().isoformat(timespec="seconds")

    payload = write_outputs(
        output_dir=output_dir,
        args=args,
        per_dataset=per_dataset,
        params=params,
        base_metadata=base_metadata,
        started_iso=started_iso,
        ended_iso=ended_iso,
        elapsed_total=elapsed_total,
    )

    summary = {
        "metadata": {**base_metadata, "cache_modes_run": ["off"]},
        "params": params,
        "aggregates_by_cache_mode": {"off": payload["aggregate"]},
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")
    print(f"\nelapsed_total: {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
