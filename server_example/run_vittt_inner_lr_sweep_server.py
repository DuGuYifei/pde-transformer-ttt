#!/usr/bin/env python3
"""Run a short ViTTT-style inner learning-rate sweep on the server.

This is intentionally a small, serial sweep:
- 128x128 training data generated under ~/working/datasets
- 20 epochs
- 1-step validation/evaluation only
- one GPU at a time

The purpose is to identify whether ViTTT-style can close the nRMSE_1 gap to
plain attention before spending time on full 100-epoch / 29-step runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_INNER_LRS = ("0.03", "0.1", "0.25", "0.5", "1.0", "2.0")


CONFIG_TEMPLATE = """# Auto-generated short ViTTT-style inner-lr sweep config.
work_dir: ~/working
data_dir: ~/working/datasets
run_root: ~/working/runs_v2
run_name: {run_name}
model_type: PDE-S
in_channels: 2
out_channels: 2
patch_size: 4
periodic: true
carrier_token_active: false

token_mixer_type: vittt
use_ttt_window_attention: false
use_ttt_state_cache_train: false
vittt_inner_lr: {inner_lr}
vittt_padding_mode: zero

learning_rate: {learning_rate}
max_epochs: {max_epochs}
batch_size: 8
num_workers: 2
devices: 1
strategy: auto
precision: 32-true
accumulate_grad_batches: 16
seed: 42

downsample_factor: 2
sample_size: 128
test_unrolling_steps: 1
max_channels: 2

resume: false
auto_resume: false
checkpoint_path: null
skip_rollout_eval: true
generate_data: false
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-lrs", nargs="+", default=list(DEFAULT_INNER_LRS))
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=str, default="4.0e-5")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--work-dir", type=Path, default=Path("~/working"))
    parser.add_argument("--data-dir", type=Path, default=Path("~/working/datasets"))
    parser.add_argument("--official-data-dir", type=Path, default=Path("~/working/datasets_official"))
    parser.add_argument("--run-root", type=Path, default=Path("~/working/runs_v2"))
    parser.add_argument("--sweep-name", default="vittt_inner_lr_128_20ep")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def resolve_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "train_ttt_ape_xxl_server.py").exists():
        return script_dir
    if (script_dir.parent / "train_ttt_ape_xxl_server.py").exists():
        return script_dir.parent
    return Path.cwd().resolve()


def lr_label(value: str) -> str:
    return value.strip().replace("-", "m").replace(".", "p")


def expand(path: Path) -> Path:
    return path.expanduser().resolve()


def expand_executable(path: Path) -> Path:
    # Do not resolve symlinks for virtualenv Python executables. Invoking the
    # venv path itself is what makes Python load that environment's site-packages.
    return path.expanduser().absolute()


def run_and_tee(cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return proc.wait()


def best_checkpoint_from_log(log_path: Path, checkpoint_dir: Path) -> Path:
    if log_path.exists():
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


def read_eval_summary(output_dir: Path) -> tuple[float | None, float | None]:
    candidates = sorted(output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        aggregate = data.get("aggregate", {})
        if not aggregate and "aggregates_by_cache_mode" in data:
            aggregate = data["aggregates_by_cache_mode"].get("off", {})
        macro = aggregate.get("macro", {})
        micro = aggregate.get("micro", {})
        macro_1 = macro.get("nRMSE_1") or macro.get("1") or macro.get(1)
        micro_1 = micro.get("nRMSE_1") or micro.get("1") or micro.get(1)
        if macro_1 is not None:
            return float(macro_1), float(micro_1) if micro_1 is not None else None
    return None, None


def main() -> int:
    args = parse_args()
    root = resolve_root()
    python = expand_executable(args.python)
    run_root = expand(args.run_root)
    work_dir = expand(args.work_dir)
    data_dir = expand(args.data_dir)
    official_data_dir = expand(args.official_data_dir)

    train_script = root / "train_ttt_ape_xxl_server.py"
    eval_script = root / "pretrained_eval" / "test_pretrained_mc_server.py"
    if not train_script.exists():
        raise FileNotFoundError(train_script)
    if not args.skip_eval and not eval_script.exists():
        raise FileNotFoundError(eval_script)

    config_dir = root / "sweep_configs" / args.sweep_name
    log_dir = root / "sweep_logs" / args.sweep_name
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results_csv = log_dir / "results.csv"
    rows: list[dict[str, str]] = []

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("PYTHONUNBUFFERED", "1")

    for inner_lr in args.inner_lrs:
        label = lr_label(inner_lr)
        run_name = f"pdes_vittt-sweep-ilr{label}_128_20ep_60sims"
        run_dir = run_root / run_name
        checkpoint_dir = run_dir / "checkpoints"
        config_path = config_dir / f"{run_name}.yaml"
        train_log = log_dir / f"{run_name}.train.log"
        eval_log = log_dir / f"{run_name}.eval.log"
        eval_dir = run_dir / "test_results" / "sweep_official_k1"

        config_path.write_text(
            CONFIG_TEMPLATE.format(
                run_name=run_name,
                inner_lr=inner_lr,
                learning_rate=args.learning_rate,
                max_epochs=args.max_epochs,
            ),
            encoding="utf-8",
        )

        if args.overwrite and run_dir.exists():
            shutil.rmtree(run_dir)

        started = time.perf_counter()
        train_cmd = [str(python), str(train_script), "--config", str(config_path)]
        train_rc = run_and_tee(train_cmd, train_log, root, env)
        train_seconds = time.perf_counter() - started

        row: dict[str, str] = {
            "run_name": run_name,
            "vittt_inner_lr": inner_lr,
            "train_returncode": str(train_rc),
            "train_seconds": f"{train_seconds:.3f}",
            "best_checkpoint": "",
            "eval_returncode": "",
            "official_macro_nRMSE_1": "",
            "official_micro_nRMSE_1": "",
        }

        if train_rc == 0:
            best_ckpt = best_checkpoint_from_log(train_log, checkpoint_dir)
            row["best_checkpoint"] = str(best_ckpt)

            if not args.skip_eval:
                eval_cmd = [
                    str(python),
                    str(eval_script),
                    "--config",
                    str(config_path),
                    "--data-dir",
                    str(official_data_dir),
                    "--checkpoint-path",
                    str(best_ckpt),
                    "--cache-mode",
                    "off",
                    "--rollout-steps",
                    "2",
                    "--eval-k",
                    "1",
                    "--output-dir",
                    str(eval_dir),
                ]
                eval_rc = run_and_tee(eval_cmd, eval_log, root, env)
                row["eval_returncode"] = str(eval_rc)
                if eval_rc == 0:
                    macro_1, micro_1 = read_eval_summary(eval_dir)
                    if macro_1 is not None:
                        row["official_macro_nRMSE_1"] = f"{macro_1:.10g}"
                    if micro_1 is not None:
                        row["official_micro_nRMSE_1"] = f"{micro_1:.10g}"

        rows.append(row)
        with results_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"[sweep] results: {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
