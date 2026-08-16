#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/venv/bin/python}"
CONFIG="$REPO_ROOT/server_example/pdes_global-linear-ttt-s-ema-3seed_256_full.yaml"

for seed in 42 43 44; do
    run_name="pdes_global-linear-ttt-s-ema_256_full_seed${seed}_100ep"
    echo "[three-seed] starting seed=$seed run_name=$run_name"
    "$PYTHON_BIN" "$REPO_ROOT/server_example/train_global_vittt_ape_xxl_server.py" \
        --config "$CONFIG" \
        --seed "$seed" \
        --run-name "$run_name"
    echo "[three-seed] completed seed=$seed"
done
