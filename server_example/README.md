# Thesis training configurations

All thesis models use `train_global_vittt_ape_xxl_server.py`. Architecture,
model scale, data profile, EMA, random seed, and hardware batch controls are
declared in YAML.

## Token mixers

| `token_mixer_type` | Behavior |
|---|---|
| `attention` | Original 8 x 8 shifted-window Attention |
| `global_linear_ttt` | Selected full-map linear PDE-TTT mixer |
| `global_vittt` | Nonlinear full-map ViT3-style mixer |
| `global_h_vittt` | H-style full-map mixer with periodic RoPE and Conv MLP |

Full-map mixers process the complete feature map at each stage. They do not
partition windows, cache fast weights, or carry state between PDE rollout
steps. `carrier_token_active` must be false for these mixers.

## Commands

```bash
# Check model construction and parameter count.
python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-linear-ttt_256_full.yaml \
  --check-config

# Check HDF5 availability and strict split identities.
python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-linear-ttt_256_full.yaml \
  --check-data

# Train or automatically resume the same run.
python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-linear-ttt_256_full.yaml
```

Use command-line overrides for a different machine without editing the
recorded YAML, for example `--devices`, `--batch-size`,
`--accumulate-grad-batches`, `--run-root`, or `--seed`.

The Full-256 configurations preserve effective global batch 128 on two GPUs.
PDE-S uses batch 8 and accumulation 8; PDE-B uses batch 2 and accumulation 32.

See [the complete thesis code guide](../docs/thesis-extension.md) for the
experiment map and implementation details.
