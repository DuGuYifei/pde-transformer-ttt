#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$HOME/server_15_global_linear_full256}"
export CONFIG="${CONFIG:-$REPO_ROOT/server_example/pdes_attention_256_full.yaml}"
export CHECKPOINT="${CHECKPOINT:-$HOME/working/runs_attention_full256/pdes_attention_s_256_full_100ep/checkpoints/epoch-epoch=099.ckpt}"
export RESULT_ROOT="${RESULT_ROOT:-$HOME/working/eval_attention_full256}"
export GPU_ID="${GPU_ID:-0}"
export BATCH_SIZE="${BATCH_SIZE:-8}"

exec bash "$REPO_ROOT/pretrained_eval/run_full256_gl_testsets_server.sh"
