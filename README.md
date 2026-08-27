# PDE-TTT Thesis Evidence

This branch retains the machine-readable evidence used by the PDE-TTT thesis. It is intentionally narrower than the local experiment history: checkpoints, HDF5 datasets, notebooks, temporal/R29 experiments, and superseded architecture variants are not versioned here.

The evidence set includes raw per-PDE and per-trajectory evaluations, training logs and metrics, derived summaries, controlled efficiency measurements, and the scripts or intermediate data used to generate thesis figures. Final rendered figures remain in the thesis repository.

## Code snapshots

| Scope | Branch | Commit |
|---|---|---|
| Final full-map linear PDE-TTT and Attention baseline | `thesis/pde-ttt` / `main` | `5b3bf08` |
| Full-map ViT3 architecture screen | `thesis/full-map-vit3` | `e3059c6` |
| Window TTT architecture screen | `thesis/window-ttt` | `a11a1a8` |
| Window Attention-TTT Hybrid | `thesis/window-attention-ttt-hybrid` | `efef9c8` |

## Evidence map

### Architecture screen at 128 resolution

| Thesis model | Evidence directory |
|---|---|
| PDE-Transformer-S | `architecture-screen-baseline/` |
| Window TTT-Linear | `architecture-screen-window-ttt-linear/` |
| Window TTT-MLP | `architecture-screen-window-ttt-mlp/` |
| Window Sequential TTT-Linear and TTT-MLP | `architecture-screen-window-sequential/` |
| Window ViT3 | `architecture-screen-window-vit3/` |
| Window Attention-TTT Hybrid | `architecture-screen-window-attention-ttt-hybrid/` |
| Full-map ViT3 | `architecture-screen-full-map-vit3/` |
| PDE-TTT-S | `architecture-screen-pde-ttt/` |

Each directory retains the evaluation summary and per-PDE results. Training logs or metrics are retained where they were available. The ordinary non-hierarchical full-map ViT3 experiment is not part of the thesis architecture screen and is intentionally absent.

### Full-256 studies

| Thesis analysis | Evidence directory |
|---|---|
| Matched PDE-Transformer-S and PDE-TTT-S | `global-linear-full256-evaluation/` |
| Focused raw/EMA comparison | `global-linear-ema-full256-evaluation/` |
| Three-seed EMA checkpoints and summaries | `global-linear-s-ema-3seed-full256-evaluation/` |
| Three-seed raw-best checkpoints and summaries | `global-linear-s-rawbest-3seed-full256-evaluation/` |
| PDE-TTT-B, raw/EMA, and published PDE-B evaluation | `global-linear-b-ema-full256-evaluation/` |
| Controlled training-step time and memory | `efficiency-benchmarks/p-vs-global-linear-ttt/` |

Evaluation directories use the following recurring files:

- `results_cache_off.csv`: per-PDE nRMSE at retained rollout horizons.
- `results_cache_off.json`: the same evaluation with complete metadata.
- `results_trajectories_cache_off.csv`: per-trajectory values when retained.
- `results_conditions_cache_off.csv`: parameter-shift aggregates by condition.
- `summary.json`: protocol, checkpoint provenance, parameter count, timing, and aggregate values.
- `*.log` and `metrics.csv`: training or evaluation provenance.

### Figure inputs

`figure-sources/` contains:

- the result-figure generation script;
- derived all-29-step and training-curve CSV inputs;
- the export and plotting scripts for qualitative autoregressive fields;
- the retained rollout arrays used to render the 16 per-family appendix figures.

The plotting script was copied from the thesis source tree to preserve the exact figure logic. Its original relative paths refer to the sibling `train-history` and thesis directories; use the evidence map above when running it from this snapshot.

## Excluded local material

The local working directory still contains earlier exploratory runs, temporal TTT/R29 experiments, checkpoints, generated HDF5 data, notebooks, caches, and thesis drafts. They are excluded from this branch because they do not support claims retained in the final thesis or are too large for a result-evidence snapshot.

