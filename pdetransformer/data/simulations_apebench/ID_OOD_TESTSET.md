# APEBench ID/OOD test set

This generator creates a test-only matrix for comparing Plain Attention and
Global Linear TTT under unseen initial-condition seeds and controlled physical
parameter shifts.

## Protocol

- Solve every PDE on the official `2048 x 2048` grid.
- Save each completed frame after non-overlapping `8 x 8` arithmetic-mean
  pooling to `256 x 256`.
- Generate 30 saved frames, corresponding to 29 autoregressive transitions.
- Generate three conditions (`id`, `ood_low`, `ood_high`) and three unseen
  seeds per PDE: nine trajectories for each of 17 PDEs.
- Define near-OOD parameters at five percent beyond the varied training
  boundary. Gray-Scott varies feed rate by minus/plus five percent because
  each named variant has a single nominal parameter pair.
- Reuse a seed across the three conditions so that the initial condition is
  held fixed while the physical parameter changes.
- Store the complete protocol in `manifest.json` and the realized parameters
  in each per-PDE JSON/HDF5 file.

The model evaluator applies its existing factor-two average pooling, producing
the `128 x 128` inputs used by the P and G-L experiments.

## Command

Run the file directly so dataset generation does not import the model training
stack:

```bash
python pdetransformer/data/simulations_apebench/generate_id_ood_testset.py \
  --output-dir ~/working/datasets_test \
  --gpu-id 0 \
  --pdes diff hyp burgers
```

The command is resumable: simulations already present as `sims/simN` are
skipped. Separate processes may generate disjoint PDE lists on different GPUs.
