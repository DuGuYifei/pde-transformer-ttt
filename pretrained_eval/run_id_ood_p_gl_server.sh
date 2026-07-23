#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/server_09_global_linear_ttt}"
PYTHON="${PYTHON:-$HOME/venv/bin/python}"
DATA_DIR="${DATA_DIR:-$HOME/working/datasets_test}"
RESULT_ROOT="${RESULT_ROOT:-$HOME/working/eval_id_ood_128/p_vs_gl}"
GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"

P_CONFIG="$REPO_ROOT/server_example/pdes_attention_128_100ep_60sims.yaml"
P_CHECKPOINT="$HOME/working/runs_global_vittt/pdes_attention_128_60sims_100ep/checkpoints/epoch-epoch=096.ckpt"
GL_CONFIG="$REPO_ROOT/server_example/pdes_global-linear-ttt_128_60sims.yaml"
GL_CHECKPOINT="$HOME/working/runs_global_vittt/pdes_global-linear-ttt_128_60sims_100ep/checkpoints/epoch-epoch=094.ckpt"
EVALUATOR="$REPO_ROOT/pretrained_eval/test_pretrained_mc_server.py"

mkdir -p "$RESULT_ROOT/plain_attention" "$RESULT_ROOT/global_linear_ttt"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

echo "Starting P at $(date --iso-8601=seconds)"
"$PYTHON" "$EVALUATOR" \
  --config "$P_CONFIG" \
  --checkpoint-path "$P_CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --id-ood-test \
  --batch-size "$BATCH_SIZE" \
  --cache-mode off \
  --output-dir "$RESULT_ROOT/plain_attention"

echo "Starting G-L at $(date --iso-8601=seconds)"
"$PYTHON" "$EVALUATOR" \
  --config "$GL_CONFIG" \
  --checkpoint-path "$GL_CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --id-ood-test \
  --batch-size "$BATCH_SIZE" \
  --cache-mode off \
  --output-dir "$RESULT_ROOT/global_linear_ttt"

echo "Finished P and G-L at $(date --iso-8601=seconds)"
