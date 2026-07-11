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
- `attention_ttt`: local attention followed by a gated TTT branch inside each block.

The causal temporal experiment keeps `token_mixer_type: attention` and enables
`temporal_ttt_enabled`. It adds one global TTT-MLP after the latent attention
stage. `TemporalRolloutSupervised` carries its fast-weight state across the
physical rollout:

```text
(u_t, W_t) -> PDE spatial encoder/attention -> latent TTT -> (u_(t+1), W_(t+1))
```

`train_unrolling_steps` controls the state lifetime, while `tbptt_chunk_size`
only controls the gradient horizon. The 128 config uses 29-step state and
four-step TBPTT. `train_step_size: 29` avoids training on nearly identical
overlapping 29-step windows.

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

Two-GPU causal temporal run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python server_example/train_ttt_ape_xxl_server.py \
  --config server_example/pdes_attention-temporal-ttt-mlp_128_20ep_60sims.yaml
```

The YAMLs expect the generated training data under:

```bash
~/working/datasets
```

For official test-set evaluation, use the scripts and notes in `pretrained_eval/`
instead of adding new test scripts here.
