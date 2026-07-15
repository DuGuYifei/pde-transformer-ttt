#!/usr/bin/env python3
"""Wait for a server training process, then evaluate its best checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--process-marker", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("~/venv/bin/python"))
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--expected-max-epochs", type=int, required=True)
    return parser.parse_args()


def expand(path: Path) -> Path:
    return path.expanduser().resolve()


def expand_executable(path: Path) -> Path:
    # Keep a virtualenv's python symlink intact so Python can locate pyvenv.cfg.
    return path.expanduser().absolute()


def process_matches(pid: int, marker: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return False
    return marker in command


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch=(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot read epoch from checkpoint name: {path.name}")
    return int(match.group(1))


def validation_losses(run_dir: Path) -> dict[int, float]:
    metrics_files = sorted(
        run_dir.glob("version_*/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    losses: dict[int, float] = {}
    for metrics_file in metrics_files:
        with metrics_file.open(newline="") as handle:
            for row in csv.DictReader(handle):
                epoch = row.get("epoch", "")
                value = row.get("val/loss", "")
                if epoch and value:
                    losses[int(epoch)] = float(value)
    return losses


def select_best_checkpoint(run_dir: Path) -> tuple[Path, int, float]:
    checkpoints = [
        path
        for path in (run_dir / "checkpoints").glob("epoch-*.ckpt")
        if not path.name.startswith("last")
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found under {run_dir / 'checkpoints'}")

    losses = validation_losses(run_dir)
    candidates = []
    for checkpoint in checkpoints:
        epoch = checkpoint_epoch(checkpoint)
        if epoch in losses:
            candidates.append((losses[epoch], epoch, checkpoint))
    if not candidates:
        raise RuntimeError("No checkpoint epoch has a matching val/loss entry")

    loss, epoch, checkpoint = min(candidates)
    return checkpoint.resolve(), epoch, loss


def write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    source_dir = expand(args.source_dir)
    run_dir = expand(args.run_dir)
    train_log = expand(args.train_log)
    config = expand(args.config)
    data_dir = expand(args.data_dir)
    python = expand_executable(args.python)
    status_path = run_dir / "official_eval_watcher.json"

    while process_matches(args.pid, args.process_marker):
        write_summary(
            status_path,
            {"status": "waiting_for_training", "pid": args.pid, "updated_at": time.time()},
        )
        time.sleep(args.poll_seconds)

    time.sleep(30)
    train_text = train_log.read_text(errors="replace")
    completion_marker = f"max_epochs={args.expected_max_epochs}` reached"
    alternate_marker = f"max_epochs={args.expected_max_epochs} reached"
    if completion_marker not in train_text and alternate_marker not in train_text:
        write_summary(
            status_path,
            {
                "status": "training_did_not_reach_expected_epoch",
                "pid": args.pid,
                "expected_max_epochs": args.expected_max_epochs,
            },
        )
        raise SystemExit(2)

    checkpoint, epoch, val_loss = select_best_checkpoint(run_dir)
    output_dir = run_dir / "test_results" / f"official_best_epoch{epoch:03d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    eval_log = output_dir / "eval.log"
    command = [
        str(python),
        "pretrained_eval/test_pretrained_mc_server.py",
        "--config",
        str(config),
        "--data-dir",
        str(data_dir),
        "--checkpoint-path",
        str(checkpoint),
        "--cache-mode",
        "auto",
        "--rollout-steps",
        "30",
        "--eval-k",
        "1",
        "10",
        "20",
        "29",
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    write_summary(
        status_path,
        {
            "status": "evaluating",
            "checkpoint": str(checkpoint),
            "best_epoch": epoch,
            "best_val_loss": val_loss,
            "output_dir": str(output_dir),
            "command": command,
        },
    )
    with eval_log.open("w") as handle:
        result = subprocess.run(
            command,
            cwd=source_dir,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    write_summary(
        status_path,
        {
            "status": "complete" if result.returncode == 0 else "evaluation_failed",
            "returncode": result.returncode,
            "checkpoint": str(checkpoint),
            "best_epoch": epoch,
            "best_val_loss": val_loss,
            "output_dir": str(output_dir),
            "eval_log": str(eval_log),
        },
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
