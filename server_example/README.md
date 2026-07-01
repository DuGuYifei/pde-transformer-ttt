# Server Example

This folder is for server-side training only. Official evaluation and pretrained-model
comparison now live under `pretrained_eval/`.

## Files

- `train_ttt_ape_xxl_server.py`: training entrypoint for PDE-S/B/L experiments.
- `smoke_2gpu.py`: CUDA/Lightning smoke test.
- `pdes_*.yaml`: training configs named as `pdes_<mixer>-<cache>_<resolution>_<sims>.yaml`.

## Mixers

The training script accepts an explicit `token_mixer_type`:

- `attention`: original PDE Transformer local window attention.
- `ttt_sequence`: the earlier sequence-style TTT window mixer.
- `vittt`: ViT^3-style local TTT mixer replacing the window attention block.

`use_ttt_window_attention` is still supported for old configs. If `token_mixer_type`
is omitted, `use_ttt_window_attention: false` maps to `attention` and `true` maps
to `ttt_sequence`.

`vittt` has no cross-step cache state. Use only `use_ttt_state_cache_train: false`
for that mixer.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python server_example/train_ttt_ape_xxl_server.py \
  --config server_example/pdes_vittt-cacheoff_128_60sims.yaml
```

The YAMLs expect the generated training data under:

```bash
~/working/datasets
```

For official test-set evaluation, use the scripts and notes in `pretrained_eval/`
instead of adding new test scripts here.

