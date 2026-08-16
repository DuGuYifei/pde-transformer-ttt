#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/server_15_global_linear_full256}"
PYTHON="${PYTHON:-$HOME/venv/bin/python}"
CONFIG="${CONFIG:-$REPO_ROOT/server_example/pdes_global-linear-ttt_256_full.yaml}"
CHECKPOINT="${CHECKPOINT:-$HOME/working/runs_global_linear_full256/pdes_global-linear-ttt_s_256_full_100ep/checkpoints/epoch-epoch=094.ckpt}"
FULL_DATA_DIR="${FULL_DATA_DIR:-$HOME/working/datasets_ape2d_full}"
OOD_DATA_DIR="${OOD_DATA_DIR:-$HOME/working/datasets_test}"
RESULT_ROOT="${RESULT_ROOT:-$HOME/working/eval_global_linear_full256}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVALUATOR="$REPO_ROOT/pretrained_eval/test_pretrained_mc_server.py"

OOD_DATASETS=(
  diff hyp burgers kdv ks fisher
  gs_alpha gs_beta gs_gamma gs_delta gs_epsilon gs_theta gs_iota gs_kappa
  sh decay_turb kolm_flow
)

mkdir -p "$RESULT_ROOT/full_test" "$RESULT_ROOT/id_ood"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

echo "Starting strict full-paper test at $(date --iso-8601=seconds)"
"$PYTHON" "$EVALUATOR" \
  --config "$CONFIG" \
  --checkpoint-path "$CHECKPOINT" \
  --data-dir "$FULL_DATA_DIR" \
  --dataset-profile full_paper \
  --strict-test-split \
  --batch-size "$BATCH_SIZE" \
  --cache-mode off \
  --output-dir "$RESULT_ROOT/full_test" \
  2>&1 | tee "$RESULT_ROOT/full_test.log"

echo "Starting ID/OOD test at $(date --iso-8601=seconds)"
"$PYTHON" "$EVALUATOR" \
  --config "$CONFIG" \
  --checkpoint-path "$CHECKPOINT" \
  --data-dir "$OOD_DATA_DIR" \
  --dataset-profile legacy_small \
  --id-ood-test \
  --datasets "${OOD_DATASETS[@]}" \
  --batch-size "$BATCH_SIZE" \
  --cache-mode off \
  --output-dir "$RESULT_ROOT/id_ood" \
  2>&1 | tee "$RESULT_ROOT/id_ood.log"

echo "Finished both evaluations at $(date --iso-8601=seconds)"
