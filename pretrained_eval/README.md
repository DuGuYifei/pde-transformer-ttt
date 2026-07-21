# Strict official test-only evaluation

The strict protocol evaluates a checkpoint on held-out simulations from all 17
PDEs in the small official `pde-transformer-ape2d` dataset. It does not train or
update the model.

## Fixed test split

| Requested PDE | Source | Source shape | Test simulations | Rollouts |
|---|---|---|---|---:|
| `diff` | `diff.hdf5` | 60 sims x 30 frames | `sim50..59` | 10 |
| `hyp` | `hyp.hdf5` | 60 sims x 30 frames | `sim50..59` | 10 |
| `burgers` | `burgers.hdf5` | 60 sims x 30 frames | `sim50..59` | 10 |
| `kdv` | `kdv.hdf5` | 60 sims x 30 frames | `sim50..59` | 10 |
| `ks` | `ks_test.hdf5` | 5 sims x 200 frames | all (`sim0..4`) | 5 |
| `fisher` | `fisher.hdf5` | 60 sims x 30 frames | `sim50..59` | 10 |
| `gs_alpha` | `gs_alpha_test.hdf5` | 3 sims x 100 frames | all (`sim0..2`) | 3 |
| `gs_beta` | `gs_beta_test.hdf5` | 3 sims x 100 frames | all (`sim0..2`) | 3 |
| `gs_gamma` | `gs_gamma_test.hdf5` | 3 sims x 100 frames | all (`sim0..2`) | 3 |
| `gs_delta` | `gs_delta.hdf5` | 100 sims x 30 frames | `sim80..99` | 20 |
| `gs_epsilon` | `gs_epsilon_test.hdf5` | 3 sims x 100 frames | all (`sim0..2`) | 3 |
| `gs_theta` | `gs_theta.hdf5` | 100 sims x 30 frames | `sim80..99` | 20 |
| `gs_iota` | `gs_iota.hdf5` | 100 sims x 30 frames | `sim80..99` | 20 |
| `gs_kappa` | `gs_kappa.hdf5` | 100 sims x 30 frames | `sim80..99` | 20 |
| `sh` | `sh.hdf5` | 60 sims x 30 frames | `sim50..59` | 10 |
| `decay_turb` | `decay_turb_test.hdf5` | 5 sims x 200 frames | all (`sim0..4`) | 5 |
| `kolm_flow` | `kolm_flow_test.hdf5` | 5 sims x 200 frames | all (`sim0..4`) | 5 |

This produces 167 complete 29-step test rollouts. Longer dedicated test files
are trimmed to one fixed rollout per simulation so each simulation has equal
weight within its PDE.

`--strict-official-test` rejects unsupported datasets, incorrect source files,
incorrect simulation IDs, unexpected frame counts, partial evaluation,
non-29-step rollouts, and data directories that do not look official. It also
records full split provenance in every JSON and CSV result.

## P command

```bash
~/venv/bin/python pretrained_eval/test_pretrained_mc_server.py \
  --config server_example/pdes_attention_128_100ep_60sims.yaml \
  --checkpoint-path ~/working/runs_global_vittt/pdes_attention_128_60sims_100ep/checkpoints/epoch-epoch=096.ckpt \
  --data-dir ~/working/datasets_official \
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
  --strict-official-test \
  --cache-mode off \
  --batch-size 8 \
  --output-dir ~/working/official_test_only_128/g-l
```

Omitting `--datasets` intentionally selects all 17 reviewed PDEs. The primary
aggregate is macro nRMSE across the 17 PDEs at rollout steps 1, 10, 20, and 29.
Micro nRMSE is supplementary because the PDEs contain different numbers of
held-out simulations.

After both runs, preserve `summary.json`, `results_cache_off.json`, and
`results_cache_off.csv`. The human-readable comparison belongs in
`train-history/p_vs_gl_official_test_only_128.md`.
