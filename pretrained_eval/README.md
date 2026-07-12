# Unified official-data evaluation

`test_pretrained_mc_server.py` is the single evaluation entry point for the
official-data report. It supports both model formats we use:

- local Lightning checkpoints trained from `server_example/*.yaml`;
- official PDE-Transformer `safetensors` loaded by `PDETransformer.from_pretrained`.

The output schema matches the previous official-eval files:

- `results_cache_off.json` / `results_cache_off.csv`
- optional `results_cache_on.json` / `results_cache_on.csv`
- `summary.json`

By default the script reports nRMSE at `k=1,10,20,29`, so the final evaluated
step is included.

## Important data rule

The YAML files in `server_example/` describe training runs, and their `data_dir`
usually points to `~/working/datasets`. For the official report, explicitly pass
the official test directory:

```bash
--data-dir ~/working/datasets_official
```

Do not rely on the YAML `data_dir` when writing numbers into
`train-history/official_data_eval_report.md`.

## Cache modes

- `attention`: cache off only.
- `vittt`: ViTTT-style cache off only. `PDEViTTTWindowBlock` has no cross-step state cache.
- `ttt_sequence`: `--cache-mode auto` runs both cache off and cache on.
- causal temporal TTT: `--cache-mode auto` evaluates persistent state only.
- official `from_pretrained` safetensors: cache off only.

Use `--cache-mode off`, `--cache-mode on`, or `--cache-mode both` to override the
auto behavior for sequence-style TTT checkpoints.

## Local checkpoint from a server YAML

Example for the ViTTT-style 128 run:

```bash
source ~/venv/bin/activate
cd ~/server_06_vittt

CUDA_VISIBLE_DEVICES=0 python pretrained_eval/test_pretrained_mc_server.py \
    --config server_example/pdes_vittt-cacheoff_128_60sims.yaml \
    --data-dir ~/working/datasets_official \
    --checkpoint-path ~/working/runs_v2/pdes_vittt-cacheoff_128_60sims/checkpoints/epoch-epoch=099.ckpt \
    --cache-mode off \
    --output-dir ~/working/runs_v2/pdes_vittt-cacheoff_128_60sims/test_results/official_best
```

The same script supports all current `server_example/*.yaml` files, including
plain attention, sequence TTT linear/MLP, ViTTT-style, causal temporal TTT,
cache-on/off configs, 128, and 256.

Quick smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python pretrained_eval/test_pretrained_mc_server.py \
    --config server_example/pdes_vittt-cacheoff_128_60sims.yaml \
    --data-dir ~/working/datasets_official \
    --checkpoint-path ~/working/runs_v2/pdes_vittt-cacheoff_128_60sims/checkpoints/epoch-epoch=099.ckpt \
    --datasets burgers \
    --max-batches-per-dataset 2 \
    --cache-mode off
```

To isolate temporal residual effects from backbone drift, load the full
checkpoint and zero only the temporal TTT residual gate:

```bash
CUDA_VISIBLE_DEVICES=0 python pretrained_eval/test_pretrained_mc_server.py \
    --config server_example/pdes_attention-temporal-ttt-mlp_128_100ep_60sims.yaml \
    --data-dir ~/working/datasets_official \
    --checkpoint-path ~/working/runs_v3/pdes_attention-temporal-ttt-mlp_128_100ep_60sims/checkpoints/epoch-epoch=087.ckpt \
    --bypass-temporal-ttt \
    --rollout-steps 2 \
    --eval-k 1
```

## Official safetensors

The official model has no Lightning `.ckpt`; load it with `from_pretrained`.
If the compute node has no internet, download on the login node first:

```bash
source ~/venv/bin/activate
hf download thuerey-group/pde-transformer --include "mc-s/*" \
    --local-dir ~/working/pretrained_weights/pde-transformer
```

Then evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 python pretrained_eval/test_pretrained_mc_server.py \
    --model-source ~/working/pretrained_weights/pde-transformer \
    --subfolder mc-s \
    --data-dir ~/working/datasets_official \
    --output-dir ~/working/runs_v2/pretrained_mc-s/test_results/official
```

If `--model-source` already points at the leaf folder containing
`config.json` and `diffusion_pytorch_model.safetensors`, pass `--subfolder ""`.

## Notes

The official pretrained model was trained on the original full-resolution
APEBench distribution. Evaluating it on our official test folder is useful as a
baseline, but it is not the same as the paper's original benchmark setting.
