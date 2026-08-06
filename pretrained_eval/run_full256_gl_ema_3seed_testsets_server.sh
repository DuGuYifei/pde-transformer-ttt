#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/server_18_global_linear_s_ema_3seed_full256}"
PYTHON="${PYTHON:-$HOME/venv/bin/python}"
CONFIG="${CONFIG:-$REPO_ROOT/server_example/pdes_global-linear-ttt-s-ema-3seed_256_full.yaml}"
RUN_ROOT="${RUN_ROOT:-$HOME/working/runs_global_linear_s_ema_3seed_full256}"
FULL_DATA_DIR="${FULL_DATA_DIR:-$HOME/working/datasets_ape2d_full}"
OOD_DATA_DIR="${OOD_DATA_DIR:-$HOME/working/datasets_test}"
RESULT_ROOT="${RESULT_ROOT:-$HOME/working/eval_global_linear_s_ema_3seed_full256}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-ema-best.ckpt}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVALUATOR="$REPO_ROOT/pretrained_eval/test_pretrained_mc_server.py"

SEEDS=(42 43 44)
EVAL_K=($(seq 1 29))
OOD_DATASETS=(
  diff hyp burgers kdv ks fisher
  gs_alpha gs_beta gs_gamma gs_delta gs_epsilon gs_theta gs_iota gs_kappa
  sh decay_turb kolm_flow
)

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

for seed in "${SEEDS[@]}"; do
  run_name="pdes_global-linear-ttt-s-ema_256_full_seed${seed}_100ep"
  checkpoint="$RUN_ROOT/$run_name/checkpoints/$CHECKPOINT_NAME"
  seed_root="$RESULT_ROOT/seed${seed}"

  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint for seed $seed: $checkpoint" >&2
    exit 1
  fi

  mkdir -p "$seed_root/full_test" "$seed_root/id_ood"

  echo "[eval] seed=$seed strict full-paper test started at $(date --iso-8601=seconds)"
  "$PYTHON" "$EVALUATOR" \
    --config "$CONFIG" \
    --checkpoint-path "$checkpoint" \
    --data-dir "$FULL_DATA_DIR" \
    --dataset-profile full_paper \
    --strict-test-split \
    --batch-size "$BATCH_SIZE" \
    --cache-mode off \
    --eval-k "${EVAL_K[@]}" \
    --output-dir "$seed_root/full_test" \
    2>&1 | tee "$seed_root/full_test.log"

  echo "[eval] seed=$seed ID/OOD test started at $(date --iso-8601=seconds)"
  "$PYTHON" "$EVALUATOR" \
    --config "$CONFIG" \
    --checkpoint-path "$checkpoint" \
    --data-dir "$OOD_DATA_DIR" \
    --dataset-profile legacy_small \
    --id-ood-test \
    --datasets "${OOD_DATASETS[@]}" \
    --batch-size "$BATCH_SIZE" \
    --cache-mode off \
    --eval-k "${EVAL_K[@]}" \
    --output-dir "$seed_root/id_ood" \
    2>&1 | tee "$seed_root/id_ood.log"

  echo "[eval] completed seed=$seed at $(date --iso-8601=seconds)"
done

echo "[eval] completed all seeds at $(date --iso-8601=seconds)"
