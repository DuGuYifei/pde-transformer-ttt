# PDE-TTT training configurations

PDE-TTT and matched Attention baselines use
`train_global_vittt_ape_xxl_server.py`. Architecture, model scale, data
profile, EMA, random seed, and hardware batch controls are declared in YAML.

| `token_mixer_type` | Behavior |
|---|---|
| `attention` | Original 8 x 8 shifted-window Attention |
| `global_linear_ttt` | Full-map linear PDE-TTT mixer |

The full-map mixer processes the complete feature map at each stage. It does
not partition windows, cache fast weights, or carry state between PDE rollout
steps. `carrier_token_active` must be false.

```bash
# Check model construction and parameter count.
python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-linear-ttt_256_full.yaml \
  --check-config

# Check HDF5 availability and disjoint simulation identities.
python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-linear-ttt_256_full.yaml \
  --check-data

# Train or resume the same run.
python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-linear-ttt_256_full.yaml
```

Use command-line overrides for a different machine without editing the
recorded YAML, for example `--devices`, `--batch-size`,
`--accumulate-grad-batches`, `--run-root`, or `--seed`.

The Full-256 configurations preserve effective global batch 128 on two GPUs.
PDE-S uses batch 8 and accumulation 8; PDE-B uses batch 2 and accumulation 32.
See the repository [README](../README.md) for data generation and evaluation.
