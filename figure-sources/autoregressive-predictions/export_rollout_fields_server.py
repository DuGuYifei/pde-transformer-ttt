"""Export one strict Full-256 rollout per PDE for qualitative comparison.

The script reuses the reviewed model and data-loading helpers from
``test_pretrained_mc_server.py``. It saves only selected time steps from one
fixed trajectory per PDE; it does not regenerate simulation data.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import test_pretrained_mc_server as evaluator


DEFAULT_STEPS = (0, 5, 10, 15, 20, 25, 29)


def parse_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--field-output-dir", type=Path, required=True)
    wrapper.add_argument("--model-label", required=True)
    wrapper.add_argument("--trajectory-index", type=int, default=0)
    wrapper.add_argument("--visualization-steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    export_args, evaluator_argv = wrapper.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *evaluator_argv]
        model_args = evaluator.parse_args()
    finally:
        sys.argv = original_argv
    return export_args, model_args


def _expand(path: Path | None) -> Path | None:
    return path.expanduser().resolve() if path is not None else None


def _select_batch(loader: Any, trajectory_index: int) -> dict[str, Any]:
    if trajectory_index < 0:
        raise ValueError("--trajectory-index must be non-negative")
    try:
        return next(itertools.islice(iter(loader), trajectory_index, trajectory_index + 1))
    except StopIteration as exc:
        raise IndexError(
            f"Trajectory index {trajectory_index} is outside a test split of length {len(loader.dataset)}."
        ) from exc


def _select_channel(reference: np.ndarray) -> int:
    """Choose the active channel with the strongest temporal change."""
    if reference.ndim != 4:
        raise ValueError(f"Expected reference [T,C,H,W], received {reference.shape}")
    temporal_change = np.mean((reference[-1] - reference[0]) ** 2, axis=(1, 2))
    temporal_variance = np.mean(np.var(reference, axis=0), axis=(1, 2))
    return int(np.argmax(temporal_change + temporal_variance))


def _load_strategy(args: argparse.Namespace, device: torch.device):
    checkpoint_path = _expand(args.checkpoint_path)
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    strategy = (
        evaluator.build_pretrained_strategy(args)
        if checkpoint_path is None
        else evaluator.build_checkpoint_strategy(args, checkpoint_path)
    )
    strategy = strategy.to(device)
    strategy.eval()
    return strategy, checkpoint_path


def main() -> None:
    export_args, args = parse_args()
    if args.dataset_profile != "full_paper":
        raise SystemExit("Qualitative Full-256 export requires --dataset-profile full_paper.")
    if args.test_unrolling_steps != 29:
        raise SystemExit("Qualitative export requires --test-unrolling-steps 29.")
    if args.sample_size != 256 or args.downsample_factor != 1:
        raise SystemExit("Qualitative export requires 256x256 inputs without downsampling.")

    steps = sorted(set(export_args.visualization_steps))
    if not steps or steps[0] < 0 or steps[-1] > 29:
        raise SystemExit("Visualization steps must be within 0..29.")

    data_dir = _expand(args.data_dir)
    if data_dir is None or not data_dir.exists():
        raise SystemExit(f"Full-256 data directory not found: {data_dir}")

    output_dir = export_args.field_output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    strategy, checkpoint_path = _load_strategy(args, device)

    datasets = list(args.datasets or evaluator.dataset_names_for_profile("full_paper"))
    expected = list(evaluator.dataset_names_for_profile("full_paper"))
    if datasets != expected:
        raise SystemExit(f"Expected the reviewed 16-PDE profile {expected}, received {datasets}.")

    records: list[dict[str, Any]] = []
    for pde in datasets:
        dm = evaluator.build_data_module(
            data_dir=data_dir,
            dataset_names=[pde],
            batch_size=1,
            num_workers=args.num_workers,
            downsample_factor=args.downsample_factor,
            test_unrolling_steps=args.test_unrolling_steps,
            max_channels=args.max_channels,
            dataset_profile=args.dataset_profile,
        )
        dm.setup(stage="test")
        split_info = evaluator.inspect_test_split(dm, pde)
        evaluator.validate_profile_test_split(pde, args.dataset_profile, split_info)
        batch = _select_batch(dm.test_dataloader(), export_args.trajectory_index)

        with torch.no_grad():
            prediction, reference = strategy.predict(batch, device=device, num_frames=30)
        prediction = np.asarray(prediction)[0]
        reference = np.asarray(reference)[0]
        if prediction.shape != reference.shape or prediction.shape[0] != 30:
            raise RuntimeError(
                f"Unexpected rollout shapes for {pde}: prediction={prediction.shape}, "
                f"reference={reference.shape}."
            )
        if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
            raise RuntimeError(f"Non-finite rollout values detected for {pde}.")

        channel = _select_channel(reference)
        payload = {
            "steps": np.asarray(steps, dtype=np.int16),
            "prediction": prediction[steps, channel].astype(np.float32),
            "reference": reference[steps, channel].astype(np.float32),
            "channel": np.asarray(channel, dtype=np.int16),
            "trajectory_index": np.asarray(export_args.trajectory_index, dtype=np.int32),
        }
        np.savez_compressed(output_dir / f"{pde}.npz", **payload)
        records.append(
            {
                "pde": pde,
                "trajectory_index": export_args.trajectory_index,
                "channel": channel,
                "steps": steps,
                "source_file": split_info["source_file_name"],
                "selected_test_trajectories": split_info["selected_num_simulations"],
                "prediction_shape": list(payload["prediction"].shape),
            }
        )
        print(f"[{export_args.model_label}] {pde}: channel={channel} shape={payload['prediction'].shape}")

        del dm, batch, prediction, reference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "model_label": export_args.model_label,
        "model_source": "from_pretrained" if checkpoint_path is None else "checkpoint",
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "hf_model_source": args.model_source if checkpoint_path is None else None,
        "subfolder": args.subfolder if checkpoint_path is None else None,
        "data_dir": str(data_dir),
        "dataset_profile": args.dataset_profile,
        "sample_size": args.sample_size,
        "rollout_steps": 30,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
