#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p train_logs

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

exec /home/yifeiliu/venv/bin/python \
  server_example/train_global_linear_rollout_server.py \
  --config server_example/pdes_global-linear-r29-persistent_128_100ep_60sims.yaml \
  >> train_logs/global_linear_r29_persistent_128_100ep.log 2>&1
