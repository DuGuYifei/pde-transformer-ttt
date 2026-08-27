# Window TTT configurations

All window experiments use
`train_global_vittt_ape_xxl_server.py`. The selected YAML records the mixer,
update schedule, training data profile, optimizer, random seed, and effective
batch settings.

```bash
python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_window-linear-ttt_128_60sims.yaml \
  --check-config

python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_window-linear-ttt_128_60sims.yaml
```

The entrypoint resumes from the last checkpoint by default. Machine-specific
settings can be overridden with `--devices`, `--batch-size`,
`--accumulate-grad-batches`, and `--run-root`.
