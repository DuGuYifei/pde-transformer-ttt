# G-L+EMA PDE-S Full-256 Evaluation

## Scope

This report isolates the effect of exponential moving average (EMA) weights
within one matched G-L+EMA training run. It compares:

- **G-L+EMA (raw weights):** raw model weights stored at the epoch selected by
  EMA validation.
- **G-L+EMA (EMA weights):** the exponential moving average exported from the
  same selected epoch.

Both variants come from the same 100-epoch PDE-S Full-256 run:

- branch: `experiment/full256-ema-controls-v1`
- implementation commit: `17d5547`
- training data: `pde-transformer-ape2d-full`
- resolution: 256 x 256
- token mixer: Global Linear TTT
- EMA decay: 0.999
- raw checkpoint: `epoch-epoch=099.ckpt`
- EMA checkpoint: `ema-best.ckpt`

## Evaluation Status

| Evaluation | Status |
|---|---|
| Raw weights, Full strict test | Complete |
| EMA weights, Full strict test | Complete |
| Raw weights, generated ID/OOD test | Complete |
| EMA weights, generated ID/OOD test | Complete |

All evaluations use batch 8 on one GTX 1080 Ti and complete one 29-transition
autoregressive rollout per trajectory.

## Full Strict Test

The Full test uses the held-out test split from
`~/working/datasets_ape2d_full`: 16 PDEs and 850 trajectories.

| Aggregate | Weights | @1 | @10 | @20 | @29 | Time |
|---|---|---:|---:|---:|---:|---:|
| Macro | Raw | 0.053097 | 0.466786 | 0.824025 | 0.967554 | 584.4 s |
| Micro | Raw | 0.054351 | 0.428004 | 0.737778 | 0.896302 | 584.4 s |
| Macro | **EMA** | **0.041823** | **0.350419** | **0.633739** | **0.762153** | 586.9 s |
| Micro | **EMA** | **0.046188** | **0.310235** | **0.534154** | **0.652399** | 586.9 s |

## Generated ID/OOD Test

The generated test matrix uses `~/working/datasets_test`: 17 PDEs, three
parameter conditions, and three unseen seeds per condition, for 153
trajectories in total.

| Condition | Weights | @1 | @10 | @20 | @29 | Time |
|---|---|---:|---:|---:|---:|---:|
| ID | Raw | 0.049461 | 0.468324 | 0.795708 | 0.967554 | -- |
| OOD-low | Raw | 0.078851 | 0.592040 | 0.939841 | 1.048890 | -- |
| OOD-high | Raw | 0.067725 | 0.494426 | 0.855770 | 1.112120 | -- |
| All | Raw | 0.065346 | 0.518263 | 0.863773 | 1.042850 | 278.8 s |
| ID | **EMA** | **0.047325** | **0.384901** | **0.621301** | **0.781693** | -- |
| OOD-low | **EMA** | **0.075486** | **0.499781** | **0.801010** | **0.928218** | -- |
| OOD-high | **EMA** | **0.066091** | **0.428016** | **0.681520** | **0.868118** | -- |
| All | **EMA** | **0.062968** | **0.437566** | **0.701277** | **0.859343** | 275.1 s |

Macro and micro are equal in the generated test because every PDE-condition
pair has the same number of trajectories.

## Observed EMA Effect

EMA weights outperform the raw weights from the same training run at every
reported horizon:

| Evaluation | @1 reduction | @10 reduction | @20 reduction | @29 reduction |
|---|---:|---:|---:|---:|
| Full Macro nRMSE | 21.23% | 24.93% | 23.09% | 21.23% |
| Full Micro nRMSE | 15.02% | 27.52% | 27.60% | 27.21% |
| Generated All nRMSE | 3.64% | 15.57% | 18.81% | 17.60% |

Inference time is effectively unchanged, as expected: EMA changes the stored
weights but not the model architecture.

## Training Reference

- best completed epoch: 100 (`epoch-099`)
- EMA validation MSE: 0.000689785
- mean training time: approximately 24.6 minutes per epoch

## Artifacts

### Raw Weights

- [Full results CSV](raw/full_test/results_cache_off.csv)
- [Full results JSON](raw/full_test/results_cache_off.json)
- [Full summary](raw/full_test/summary.json)
- [Full evaluation log](raw/full_test.log)
- [OOD overall CSV](raw/id_ood/results_cache_off.csv)
- [OOD condition CSV](raw/id_ood/results_conditions_cache_off.csv)
- [OOD trajectory CSV](raw/id_ood/results_trajectories_cache_off.csv)
- [OOD summary](raw/id_ood/summary.json)
- [OOD evaluation log](raw/id_ood.log)

### EMA Weights

- [Full results CSV](ema/full_test/results_cache_off.csv)
- [Full results JSON](ema/full_test/results_cache_off.json)
- [Full summary](ema/full_test/summary.json)
- [Full evaluation log](ema/full_test.log)
- [OOD overall CSV](ema/id_ood/results_cache_off.csv)
- [OOD condition CSV](ema/id_ood/results_conditions_cache_off.csv)
- [OOD trajectory CSV](ema/id_ood/results_trajectories_cache_off.csv)
- [OOD summary](ema/id_ood/summary.json)
- [OOD evaluation log](ema/id_ood.log)
