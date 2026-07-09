#!/usr/bin/env python3
"""Train a 20-epoch plain attention baseline and evaluate official nRMSE_1."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path("/home/yifeiliu/venv/bin/python"))
    parser.add_argument("--config", type=Path, default=Path("pdes_attention-cacheoff_128_60sims.yaml"))
    parser.add_argument("--run-name", default="pdes_attention_128_20ep_60sims")
    parser.add_argument("--run-root", type=Path, default=Path("~/working/runs_v2"))
    parser.add_argument("--official-data-dir", type=Path, default=Path("~/working/datasets_official"))
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--max-epochs", type=int, default=20)
    return parser.parse_args()


def run_and_tee(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return proc.wait()


def best_checkpoint_from_log(log_path: Path, checkpoint_dir: Path) -> Path:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"best checkpoints:\s*(\{.*?\})", text, re.S)
    if match:
        entries = re.findall(
            r"['\"]([^'\"]+\.ckpt)['\"]\s*:\s*tensor\(([-+0-9.eE]+)",
            match.group(1),
        )
        if entries:
            best_path, _ = min(entries, key=lambda item: float(item[1]))
            return Path(best_path).expanduser().resolve()

    last = checkpoint_dir / "last.ckpt"
    if last.exists():
        return last
    candidates = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    return candidates[0]


def macro_from_json(output_dir: Path) -> tuple[float | None, float | None]:
    for path in sorted(output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        aggregate = data.get("aggregate", {})
        if not aggregate and "aggregates_by_cache_mode" in data:
            aggregate = data["aggregates_by_cache_mode"].get("off", {})
        macro = aggregate.get("macro", {})
        micro = aggregate.get("micro", {})
        if "nRMSE_1" in macro:
            return float(macro["nRMSE_1"]), float(micro["nRMSE_1"]) if "nRMSE_1" in micro else None
    return None, None


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    python = args.python.expanduser().absolute()
    config = args.config.expanduser()
    if not config.is_absolute():
        config = root / config
    run_root = args.run_root.expanduser().resolve()
    run_dir = run_root / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    output_dir = run_dir / "test_results" / "official_k1_20ep"
    log_dir = root / "sweep_logs" / "attention_128_20ep"
    train_log = log_dir / "train.log"
    eval_log = log_dir / "eval.log"
    result_json = log_dir / "result.json"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONUNBUFFERED"] = "1"

    started = time.perf_counter()
    train_cmd = [
        str(python),
        str(root / "train_ttt_ape_xxl_server.py"),
        "--config",
        str(config),
        "--run-name",
        args.run_name,
        "--max-epochs",
        str(args.max_epochs),
        "--test-unrolling-steps",
        "1",
        "--skip-rollout-eval",
        "--no-auto-resume",
    ]
    train_rc = run_and_tee(train_cmd, train_log, env)
    if train_rc != 0:
        return train_rc

    best_ckpt = best_checkpoint_from_log(train_log, checkpoint_dir)
    eval_cmd = [
        str(python),
        str(root / "pretrained_eval" / "test_pretrained_mc_server.py"),
        "--config",
        str(config),
        "--run-name",
        args.run_name,
        "--data-dir",
        str(args.official_data_dir.expanduser().resolve()),
        "--checkpoint-path",
        str(best_ckpt),
        "--cache-mode",
        "off",
        "--rollout-steps",
        "2",
        "--eval-k",
        "1",
        "--test-unrolling-steps",
        "1",
        "--output-dir",
        str(output_dir),
    ]
    eval_rc = run_and_tee(eval_cmd, eval_log, env)
    macro_1, micro_1 = macro_from_json(output_dir) if eval_rc == 0 else (None, None)
    result = {
        "run_name": args.run_name,
        "train_returncode": train_rc,
        "eval_returncode": eval_rc,
        "seconds": time.perf_counter() - started,
        "best_checkpoint": str(best_ckpt),
        "official_macro_nRMSE_1": macro_1,
        "official_micro_nRMSE_1": micro_1,
    }
    result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return eval_rc


if __name__ == "__main__":
    raise SystemExit(main())
