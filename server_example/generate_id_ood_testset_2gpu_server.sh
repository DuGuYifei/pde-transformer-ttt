#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/server_14_ape2d_id_ood_generation}"
PYTHON="${PYTHON:-$HOME/venv_apebench_gen/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/working/datasets_test}"
LOG_DIR="${LOG_DIR:-$HOME/working/datasets_test_logs}"
GENERATOR="$ROOT/pdetransformer/data/simulations_apebench/generate_id_ood_testset.py"
AUDITOR="$ROOT/pdetransformer/data/simulations_apebench/audit_id_ood_testset.py"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
"$PYTHON" "$GENERATOR" --output-dir "$OUTPUT_DIR" --write-manifest-only \
  > "$LOG_DIR/manifest.log" 2>&1
"$PYTHON" "$AUDITOR" --output-dir "$OUTPUT_DIR" \
  > "$LOG_DIR/audit_before_resume.log" 2>&1

run_queue() {
  local gpu_id="$1"
  local label="$2"
  shift 2
  local pdes=("$@")
  local failures=0

  export PYTHONUNBUFFERED=1
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
  for pde in "${pdes[@]}"; do
    printf 'PDE_PROCESS_START gpu=%s pde=%s time=%s\n' "$gpu_id" "$pde" "$(date -Is)"
    if "$PYTHON" "$GENERATOR" \
      --output-dir "$OUTPUT_DIR" --gpu-id "$gpu_id" --pdes "$pde"; then
      printf 'PDE_PROCESS_DONE gpu=%s pde=%s time=%s\n' "$gpu_id" "$pde" "$(date -Is)"
    else
      failures=$((failures + 1))
      printf 'PDE_PROCESS_FAILED gpu=%s pde=%s time=%s\n' "$gpu_id" "$pde" "$(date -Is)"
    fi
  done
  printf 'GPU_QUEUE_DONE gpu=%s label=%s failures=%s time=%s\n' \
    "$gpu_id" "$label" "$failures" "$(date -Is)"
  test "$failures" -eq 0
}

run_kolm_after_smoke() {
  export PYTHONUNBUFFERED=1
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
  printf 'KOLM_SMOKE_START sim=3 time=%s\n' "$(date -Is)"
  if "$PYTHON" "$GENERATOR" \
    --output-dir "$OUTPUT_DIR" --gpu-id 0 --pdes kolm_flow --sim-ids 3; then
    printf 'KOLM_SMOKE_DONE sim=3 time=%s\n' "$(date -Is)"
    run_queue 0 gpu0_kolm_flow kolm_flow
  else
    printf 'KOLM_SMOKE_FAILED sim=3 time=%s\n' "$(date -Is)"
    return 1
  fi
}

export ROOT PYTHON OUTPUT_DIR LOG_DIR GENERATOR AUDITOR
export -f run_queue run_kolm_after_smoke

nohup bash -c 'run_kolm_after_smoke' \
  > "$LOG_DIR/resume_gpu0_kolm_flow.log" 2>&1 < /dev/null &
gpu0_pid=$!
printf '%s\n' "$gpu0_pid" > "$LOG_DIR/resume_gpu0.pid"

nohup bash -c 'run_queue 1 gpu1_other16 \
  diff hyp burgers kdv ks fisher \
  gs_alpha gs_beta gs_gamma gs_delta gs_epsilon gs_theta gs_iota gs_kappa \
  sh decay_turb' \
  > "$LOG_DIR/resume_gpu1_other16.log" 2>&1 < /dev/null &
gpu1_pid=$!
printf '%s\n' "$gpu1_pid" > "$LOG_DIR/resume_gpu1.pid"

printf 'GPU0_PID=%s GPU1_PID=%s\n' "$gpu0_pid" "$gpu1_pid"
