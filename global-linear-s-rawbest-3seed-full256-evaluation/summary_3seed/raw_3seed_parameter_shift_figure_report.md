# Raw Three-Seed ID/OOD Figure

This figure reuses the generated 17-family ID/OOD test matrix, but replaces the
single PDE-TTT-S checkpoint in the original Figure 6.6 with the mean of three
independently trained raw-best checkpoints (model seeds 42, 43, and 44).

The PDE-Transformer-S reference remains one raw checkpoint. Therefore, this is
useful for checking PDE-TTT-S training-seed stability, but it is not a matched
three-seed comparison between both architectures.

## Files

- `raw_3seed_parameter_shift_response_matrix.png`: direct replacement candidate
  for the original mean-difference heatmap.
- `raw_3seed_parameter_shift_mean_and_consistency.png`: diagnostic figure. Its
  lower panel reports how many of the three PDE-TTT-S checkpoints beat the
  PDE-Transformer-S checkpoint in each cell.
- `raw_3seed_parameter_shift_values.csv`: all plotted values.
- `plot_raw_3seed_parameter_shift.py`: reproducible plotting script.

Blue denotes lower PDE-TTT-S nRMSE. Hyper-diffusion is outlined because it was
not present in final Full-256 training.

## Aggregate Comparison

| Condition | Step | PDE-Transformer-S raw | PDE-TTT-S raw, 3-seed mean | Relative change | Families with lower mean | Families won by all 3 seeds |
|---|---:|---:|---:|---:|---:|---:|
| ID | 1 | 0.054162 | 0.050105 | -7.5% | 13/17 | 10/17 |
| ID | 10 | 0.473960 | 0.428319 | -9.6% | 13/17 | 9/17 |
| ID | 20 | 0.737786 | 0.665956 | -9.7% | 12/17 | 8/17 |
| ID | 29 | 0.893415 | 0.777732 | -12.9% | 12/17 | 7/17 |
| OOD-low | 1 | 0.086025 | 0.079933 | -7.1% | 10/17 | 8/17 |
| OOD-low | 10 | 0.574870 | 0.515259 | -10.4% | 14/17 | 9/17 |
| OOD-low | 20 | 0.853415 | 0.781017 | -8.5% | 12/17 | 7/17 |
| OOD-low | 29 | 0.961001 | 0.877304 | -8.7% | 13/17 | 7/17 |
| OOD-high | 1 | 0.070461 | 0.068319 | -3.0% | 11/17 | 10/17 |
| OOD-high | 10 | 0.413192 | 0.480574 | +16.3% | 9/17 | 5/17 |
| OOD-high | 20 | 0.771780 | 0.691331 | -10.4% | 10/17 | 5/17 |
| OOD-high | 29 | 0.960748 | 0.833216 | -13.3% | 9/17 | 6/17 |

Across all 204 family/condition/horizon cells, the PDE-TTT-S three-seed mean is
lower in 138 cells. All three PDE-TTT-S checkpoints beat the baseline in 91
cells, while none beats it in 32 cells.

## Interpretation

The three-seed mean preserves the main aggregate result: PDE-TTT-S is better in
11 of the 12 condition/horizon aggregates, with OOD-high at step 10 as the only
exception. At step 29, the mean reduction is 12.9% for ID, 8.7% for OOD-low,
and 13.3% for OOD-high.

The diagnostic panel is more informative than a mean heatmap alone because it
shows where a blue mean is supported by all three independently trained models.
However, the original single-seed Figure 6.6 remains the cleaner controlled
architecture comparison until PDE-Transformer-S is also trained with three
independent seeds.
