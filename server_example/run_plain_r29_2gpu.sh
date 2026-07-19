#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p train_logs

CUDA_VISIBLE_DEVICES=0,1 \
PYTHONUNBUFFERED=1 \
  "$HOME/venv/bin/python" -u server_example/train_plain_rollout_server.py \
  --config server_example/pdes_attention-r29_128_100ep_60sims.yaml \
  > train_logs/plain_r29_128.log 2>&1
