# P vs G-L strict official test

The strict protocol evaluates the clean 128-resolution P and G-L checkpoints on
held-out simulations from the small official `pde-transformer-ape2d` dataset.
It does not train or update either model.

## Fixed test split

| Requested PDE | Source | Simulations | Rollouts |
|---|---|---|---:|
| `burgers` | `burgers.hdf5` | `sim50` through `sim59` | 10 |
| `ks` | `ks_test.hdf5` | `sim0` through `sim4` | 5 |
| `kolm_flow` | `kolm_flow_test.hdf5` | `sim0` through `sim4` | 5 |

`--strict-official-test` rejects unsupported datasets, partial evaluation,
non-29-step rollouts, and data directories that do not look official. It also
records the resolved source file and simulation IDs in every JSON and CSV row.

## P command

```bash
~/venv/bin/python pretrained_eval/test_pretrained_mc_server.py \
  --config server_example/pdes_attention_128_100ep_60sims.yaml \
  --checkpoint-path ~/working/runs_global_vittt/pdes_attention_128_60sims_100ep/checkpoints/epoch-epoch=096.ckpt \
  --data-dir ~/working/datasets_official \
  --datasets burgers ks kolm_flow \
  --strict-official-test \
  --cache-mode off \
  --batch-size 8 \
  --output-dir ~/working/official_test_only_128/p
```

## G-L command

```bash
~/venv/bin/python pretrained_eval/test_pretrained_mc_server.py \
  --config server_example/pdes_global-linear-ttt_128_60sims.yaml \
  --checkpoint-path ~/working/runs_global_vittt/pdes_global-linear-ttt_128_60sims_100ep/checkpoints/epoch-epoch=094.ckpt \
  --data-dir ~/working/datasets_official \
  --datasets burgers ks kolm_flow \
  --strict-official-test \
  --cache-mode off \
  --batch-size 8 \
  --output-dir ~/working/official_test_only_128/g-l
```

The primary aggregate is macro nRMSE across the three PDEs at rollout steps
1, 10, 20, and 29. Micro nRMSE is supplementary because Burgers contributes
twice as many trajectories as either KS or Kolmogorov Flow.

After both runs, preserve `summary.json`, `results_cache_off.json`, and
`results_cache_off.csv`. The human-readable comparison belongs in
`train-history/p_vs_gl_official_test_only_128.md`.
