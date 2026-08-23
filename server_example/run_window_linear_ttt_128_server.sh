#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_path="$repo_root/train_logs/window_linear_ttt_128_100ep.log"

mkdir -p "$(dirname "$log_path")"
cd "$repo_root"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1

exec > >(tee -a "$log_path") 2>&1
echo "[launch] $(date --iso-8601=seconds) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
exec /home/yifeiliu/venv/bin/python \
    server_example/train_global_vittt_ape_xxl_server.py \
    --config server_example/pdes_window-linear-ttt_128_60sims.yaml \
    "$@"
