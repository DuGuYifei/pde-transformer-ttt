# P vs G-L PDE-S Full-256 Matched Evaluation

## Scope

This report compares two PDE-S models trained from scratch under the same
Full-256 protocol:

| ID | Token mixer | Best checkpoint | Parameters |
|---|---|---|---:|
| **P** | Original 8x8 shifted-window Attention | Plain epoch-099 | 33,190,328 |
| **G-L** | Full-map Global Linear TTT | G-L epoch-094 | 33,361,952 |

G-L replaces every shifted-window Attention mixer with one non-causal,
full-feature-map Linear TTT update. It has no temporal TTT layer, no
cross-rollout fast-weight cache, and resets its temporary fast weights from
the learned `W0` on every model forward.

Both runs use:

- `pde-transformer-ape2d-full`, `dataset_profile: full_paper`
- resolution 256 x 256 and single-step training
- the same simulation split, seed, optimizer, LR schedule, and 100 epochs
- two GTX 1080 Ti GPUs, FP32, batch 8/GPU, gradient accumulation 8
- effective global batch 128
- best-checkpoint selection by validation loss

Code:

- P branch: `experiment/plain-full256-v1`
- P training/evaluation launcher: `b207d81`
- G-L branch: `experiment/global-linear-full256-v1`
- Strict Full/OOD evaluator: `8391f88`
- G-L reproducible launcher: `e6a4ce2`

## Main Results

Lower nRMSE is better. G-L is better on every Full-test aggregate and every
rollout horizon.

### Full Strict Test

The Full test contains 16 PDEs and 850 held-out trajectories.

| Aggregate | Model | @1 | @10 | @20 | @29 |
|---|---|---:|---:|---:|---:|
| Macro | P | 0.049481 | 0.467861 | 0.781651 | 0.903583 |
| Macro | **G-L** | **0.044101** | **0.373365** | **0.675741** | **0.803795** |
| Micro | P | 0.053381 | 0.405824 | 0.688354 | 0.825951 |
| Micro | **G-L** | **0.047342** | **0.330058** | **0.573417** | **0.694218** |

Relative to P, G-L reduces:

| Aggregate | @1 | @10 | @20 | @29 |
|---|---:|---:|---:|---:|
| Macro nRMSE | 10.87% | 20.20% | 13.55% | 11.04% |
| Micro nRMSE | 11.31% | 18.67% | 16.70% | 15.95% |

### Generated ID/OOD Test

The generated matrix contains 17 PDEs, three parameter conditions, and three
unseen seeds per condition: 153 trajectories total. Macro and micro are equal
because every PDE and condition has the same trajectory count.

| Condition | Model | @1 | @10 | @20 | @29 |
|---|---|---:|---:|---:|---:|
| ID | P | 0.054162 | 0.473960 | 0.737786 | 0.893413 |
| ID | **G-L** | **0.051201** | **0.405732** | **0.622333** | **0.746137** |
| OOD-low | P | 0.086025 | 0.574870 | 0.853413 | 0.961002 |
| OOD-low | **G-L** | **0.079093** | **0.498325** | **0.774820** | **0.908577** |
| OOD-high | P | 0.070461 | **0.413192** | 0.771780 | 0.960748 |
| OOD-high | **G-L** | **0.069580** | 0.480746 | **0.660825** | **0.800978** |
| All | P | 0.070216 | 0.487341 | 0.787660 | 0.938388 |
| All | **G-L** | **0.066624** | **0.461601** | **0.685993** | **0.818564** |

G-L wins 11 of the 12 condition/horizon comparisons. The exception is
OOD-high at step 10, where P is better (`0.413192` vs `0.480746`).

## Training Curves

![P vs G-L Full-256 training curves](p_vs_global_linear_training_curves.png)

The logged best validation losses are:

| Model | Best completed epoch | Best validation MSE |
|---|---:|---:|
| P | 100 (`epoch-099.ckpt`) | 0.00124780 |
| G-L | 95 (`epoch-094.ckpt`) | 0.000775277 |

G-L has lower training and validation MSE over most of training. Both curves
still fluctuate near the end, but G-L's rollout advantage is confirmed by the
independent nRMSE tests rather than inferred only from validation MSE.

## Runtime

Training time is derived from one deduplicated rank record per completed
epoch. Test time is the evaluator's directly measured wall-clock duration.

| Work | Model | Hardware | Batch | Measured time |
|---|---|---|---:|---:|
| Training, 100 epochs | P | 2 x GTX 1080 Ti | 8/GPU, effective 128 | 25.600 min/epoch; 2560.0 min total |
| Training, 100 epochs | G-L | 2 x GTX 1080 Ti | 8/GPU, effective 128 | 24.513 min/epoch; 2451.3 min total |
| Full strict test, 850 x 29 | P | 1 x GTX 1080 Ti | 8 | 580.6 s |
| Full strict test, 850 x 29 | G-L | 1 x GTX 1080 Ti | 8 | 584.5 s |
| ID/OOD test, 153 x 29 | P | 1 x GTX 1080 Ti | 8 | 277.8 s |
| ID/OOD test, 153 x 29 | G-L | 1 x GTX 1080 Ti | 8 | 275.5 s |

G-L uses 171,624 more parameters than P, an increase of 0.52%. Its logged
training epoch is 4.25% faster. Inference time is effectively tied in these
two runs: G-L is 0.7% slower on Full and 0.8% faster on ID/OOD. These small
differences should not be treated as a robust inference-speed advantage
without repeated timing runs.

## Full Test Protocol

- Data: `~/working/datasets_ape2d_full`
- Profile: `full_paper`
- Resolution: 256 x 256 (`downsample_factor=1`)
- PDEs: 16; `hyp` is not part of this profile
- Standard joint files: held-out `sim500..sim599`
- Joint Gray-Scott files: held-out `sim80..sim99`
- Separate-test datasets: all simulations in the corresponding
  `*_test.hdf5`
- One complete 29-transition autoregressive rollout per trajectory
- Strict source-file and simulation-ID validation: passed for both models

Macro first averages within each PDE and then weights all PDEs equally. Micro
averages all 850 trajectories directly.

### Full Per-PDE Comparison

| PDE | N | P @1 | G-L @1 | P @10 | G-L @10 | P @20 | G-L @20 | P @29 | G-L @29 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diff | 100 | 0.0461 | **0.0338** | 0.2671 | **0.1554** | 0.5184 | **0.2233** | 0.7874 | **0.2691** |
| burgers | 100 | 0.0273 | **0.0239** | 0.1697 | **0.1104** | 0.3594 | **0.2387** | 0.5421 | **0.3782** |
| kdv | 100 | 0.0502 | **0.0501** | 0.2296 | **0.2132** | 0.4177 | **0.3642** | 0.6070 | **0.5105** |
| ks | 50 | 0.0291 | **0.0171** | 0.4519 | **0.3125** | 1.0718 | **0.8872** | 1.3358 | **1.2183** |
| fisher | 100 | 0.0250 | **0.0225** | 0.3599 | **0.3470** | 0.6817 | **0.5721** | 0.6625 | **0.5514** |
| gs_alpha | 30 | 0.0259 | **0.0257** | 0.6198 | **0.2673** | 1.0227 | **0.7288** | 1.2342 | **1.0236** |
| gs_beta | 30 | 0.0348 | **0.0307** | 0.6223 | **0.4485** | 0.9664 | **0.6829** | 1.0720 | **0.7424** |
| gs_gamma | 30 | 0.0372 | **0.0313** | 0.5071 | **0.4179** | 0.8017 | **0.7617** | **0.9538** | 0.9639 |
| gs_delta | 20 | **0.0215** | 0.0228 | 0.7913 | **0.7113** | 1.0860 | **1.0643** | 1.0048 | **0.9935** |
| gs_epsilon | 30 | 0.0248 | **0.0168** | 0.2266 | **0.1784** | 0.4268 | **0.3151** | 0.6376 | **0.4909** |
| gs_theta | 20 | **0.0169** | 0.0175 | 0.7517 | **0.6221** | 1.0072 | **1.0036** | **0.9827** | 1.0220 |
| gs_iota | 20 | **0.0171** | 0.0176 | 0.3108 | **0.2435** | 0.7707 | **0.7412** | 0.7470 | **0.7437** |
| gs_kappa | 20 | 0.0283 | **0.0252** | 0.3771 | **0.2901** | 0.7955 | **0.6615** | 0.9666 | **0.8672** |
| sh | 100 | 0.0665 | **0.0606** | 0.5159 | **0.4562** | 0.7010 | **0.6156** | 0.7647 | **0.6958** |
| decay_turb | 50 | 0.2455 | **0.2328** | **0.7734** | 0.7915 | **0.8873** | 1.0604 | **0.9540** | 1.2182 |
| kolm_flow | 50 | 0.0953 | **0.0772** | 0.5116 | **0.4086** | 0.9921 | **0.8913** | 1.2052 | **1.1719** |

G-L wins on 13/16 PDEs at @1, 15/16 at @10, 15/16 at @20, and
13/16 at @29. At @29, P remains better on `gs_gamma`, `gs_theta`, and
`decay_turb`; the aggregate conclusion is broad but not universal.

## ID/OOD Protocol

- Data: `~/working/datasets_test`
- Resolution: generated and evaluated at 256 x 256
- PDEs: 17, including `hyp`
- Conditions:
  - `id`: parameter inside the training range
  - `ood_low`: 5% beyond the lower training boundary
  - `ood_high`: 5% beyond the upper training boundary
- Three unseen seeds per condition and nine trajectories per PDE
- Simulations were solved at 2048 x 2048, then mean-pooled 8 x 8 to 256 x 256
- Manifest/HDF5 audit: 153/153 trajectories valid
- ID: `sim0..sim2`; OOD-low: `sim3..sim5`; OOD-high: `sim6..sim8`

The evaluator uses the same mean-std normalization code path for both models,
with statistics read from the evaluated HDF5 data.

### Step-29 OOD Per-PDE Comparison

| PDE | P ID | G-L ID | P OOD-low | G-L OOD-low | P OOD-high | G-L OOD-high |
|---|---:|---:|---:|---:|---:|---:|
| diff | 0.5818 | **0.1319** | 0.6292 | **0.5379** | 0.9283 | **0.4247** |
| hyp | 1.9283 | **0.9209** | 1.4361 | **1.3736** | 2.2156 | **0.7766** |
| burgers | 0.6001 | **0.4539** | 0.6059 | **0.4659** | 0.5964 | **0.4449** |
| kdv | 0.5436 | **0.4610** | 0.5437 | **0.4611** | 0.5435 | **0.4608** |
| ks | 1.2974 | **1.1419** | **1.1560** | 1.5330 | 1.2689 | **1.0973** |
| fisher | 0.3508 | **0.2758** | 0.5159 | **0.4582** | 0.2765 | **0.2577** |
| gs_alpha | 1.1304 | **1.0270** | 1.2603 | **0.8127** | **0.9917** | 1.2209 |
| gs_beta | 1.0687 | **0.9715** | **1.0942** | 1.1540 | 2.2931 | **1.4089** |
| gs_gamma | **1.0026** | 1.0485 | 1.2695 | **1.2505** | **0.9281** | 0.9696 |
| gs_delta | **1.0049** | 1.0051 | 1.1063 | **1.1000** | 0.9552 | **0.9472** |
| gs_epsilon | 0.6630 | **0.4925** | 0.8139 | **0.5755** | **0.4944** | 0.7568 |
| gs_theta | **1.0170** | 1.0562 | **1.0701** | 1.1203 | **0.9843** | 1.0336 |
| gs_iota | 0.7544 | **0.6922** | 0.9981 | **0.9195** | **0.7141** | 0.7263 |
| gs_kappa | 0.9032 | **0.8000** | 0.9065 | **0.8922** | 1.0093 | **0.8576** |
| sh | **0.7333** | 0.7574 | 1.2317 | **1.1345** | **0.5797** | 0.7759 |
| decay_turb | 0.4151 | **0.3092** | 0.5422 | **0.4954** | 0.4316 | **0.3091** |
| kolm_flow | 1.1935 | **1.1393** | **1.1574** | 1.1616 | **1.1222** | 1.1488 |

Three seeds per condition are enough for a controlled first comparison, but
not for a tight confidence interval. The Full strict test is the stronger
source for the overall P vs G-L conclusion.

## Conclusion

Under the matched PDE-S Full-256 setup, G-L is the stronger model:

- lower Full-test macro and micro nRMSE at every evaluated horizon
- lower overall ID/OOD nRMSE at every horizon
- 4.25% lower measured training time per epoch
- essentially unchanged parameter count and single-GPU inference time

The result supports replacing PDE-S shifted-window Attention with Global
Linear TTT for this training setup. It does not show that G-L wins on every
individual PDE or every controlled parameter shift, and it does not yet
establish how the efficiency gap scales to PDE-B/PDE-L.

## Artifacts

### Comparison

- [Training curves](p_vs_global_linear_training_curves.png)
- [Parsed training metrics](p_vs_global_linear_training_metrics.csv)
- [Curve-generation script](plot_training_curves.py)

### G-L

- [Full results CSV](full_test/results_cache_off.csv)
- [Full results JSON](full_test/results_cache_off.json)
- [Full summary](full_test/summary.json)
- [OOD overall CSV](id_ood/results_cache_off.csv)
- [OOD condition CSV](id_ood/results_conditions_cache_off.csv)
- [OOD trajectory CSV](id_ood/results_trajectories_cache_off.csv)
- [OOD summary](id_ood/summary.json)
- [Full evaluation log](full_test.log)
- [OOD evaluation log](id_ood.log)
- [Training log](training.log)

### P

- [Full results CSV](plain_attention/full_test/results_cache_off.csv)
- [Full results JSON](plain_attention/full_test/results_cache_off.json)
- [Full summary](plain_attention/full_test/summary.json)
- [OOD overall CSV](plain_attention/id_ood/results_cache_off.csv)
- [OOD condition CSV](plain_attention/id_ood/results_conditions_cache_off.csv)
- [OOD trajectory CSV](plain_attention/id_ood/results_trajectories_cache_off.csv)
- [OOD summary](plain_attention/id_ood/summary.json)
- [Full evaluation log](plain_attention/full_test.log)
- [OOD evaluation log](plain_attention/id_ood.log)
- [Training log](plain_attention/training.log)
