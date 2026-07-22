#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/server_14_ape2d_id_ood_generation}"
PYTHON="${PYTHON:-$HOME/venv_apebench_gen/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/working/datasets_test}"
LOG_DIR="${LOG_DIR:-$HOME/working/datasets_test_logs}"
GENERATOR="$ROOT/pdetransformer/data/simulations_apebench/generate_id_ood_testset.py"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
"$PYTHON" "$GENERATOR" --output-dir "$OUTPUT_DIR" --write-manifest-only \
  > "$LOG_DIR/manifest.log" 2>&1

nohup env PYTHONUNBUFFERED=1 "$PYTHON" "$GENERATOR" \
  --output-dir "$OUTPUT_DIR" --gpu-id 0 --pdes kolm_flow \
  > "$LOG_DIR/gpu0_kolm_flow.log" 2>&1 < /dev/null &
gpu0_pid=$!
printf '%s\n' "$gpu0_pid" > "$LOG_DIR/gpu0.pid"

nohup env PYTHONUNBUFFERED=1 "$PYTHON" "$GENERATOR" \
  --output-dir "$OUTPUT_DIR" --gpu-id 1 \
  --pdes diff hyp burgers kdv ks fisher \
          gs_alpha gs_beta gs_gamma gs_delta gs_epsilon gs_theta gs_iota gs_kappa \
          sh decay_turb \
  > "$LOG_DIR/gpu1_other16.log" 2>&1 < /dev/null &
gpu1_pid=$!
printf '%s\n' "$gpu1_pid" > "$LOG_DIR/gpu1.pid"

printf 'GPU0_PID=%s GPU1_PID=%s\n' "$gpu0_pid" "$gpu1_pid"
