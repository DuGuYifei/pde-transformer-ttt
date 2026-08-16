# Legacy architecture-screening code

This branch preserves the spatial TTT implementations used in the early
128-resolution architecture screen. It starts from upstream PDE-Transformer
commit `850e09d` and stops at historical commit `d08e102`, before persistent
temporal TTT was introduced.

The thesis final implementation and matched Full-256 experiments live on
`main` (also published as `thesis/final`). This legacy branch is retained for
provenance and should not be used as the starting point for new PDE-TTT work.

## Models represented here

| Thesis label | Configuration | Implementation |
|---|---|---|
| Shifted-window Attention | `server_example/pdes_attention-cacheoff_128_60sims.yaml` | Original Attention path |
| Window TTT-Linear A | `server_example/pdes_ttt-sequence-linear-cacheoff_128_60sims.yaml` | `ttt_window_attention.py` |
| Window TTT-Linear B | `server_example/pdes_ttt-sequence-linear-cacheon_128_60sims.yaml` | Same learner with the historical cache flag |
| Window TTT-MLP | `server_example/pdes_ttt-mlp-cacheon_128_60sims.yaml` | MLP learner used by the recorded run |
| Early window ViT3-style | `server_example/pdes_vittt-cacheoff_128_60sims.yaml` | `pde_vittt_window.py` |
| Local Attention--TTT hybrid | `server_example/pdes_attention-ttt-mlp-bidir_128_60sims.yaml` | Attention followed by gated local TTT |

All runs use `server_example/train_ttt_ape_xxl_server.py`.

## Important interpretation limits

- Window TTT processes each 8 x 8 window as a 64-token sequence. It is not the
  selected full-map PDE-TTT architecture.
- The historical `cacheon` training label does not establish cross-PDE-step
  state. The single-step trainer recreated and discarded the cache between
  batches. The thesis therefore names the two linear runs A and B instead of
  treating B as validated temporal memory.
- The early ViT3-style implementation applies its learner independently in
  each window and predates the corrected full-map CPE, head layout, gradient
  normalization, and whole-domain periodic convolution.
- The local Attention--TTT hybrid keeps both operations inside the same
  64-token window. It is inspired by the ordering used in Video-DiT but does
  not reproduce Video-DiT's global TTT scope.
- Results from this branch are an exploratory architecture screen with a
  heterogeneous historical lineage. Final effect-size claims use the matched
  Attention and PDE-TTT code on `main`.

## Source map

```text
pdetransformer/core/ttt_window_attention.py
    Sequential TTT-Linear and TTT-MLP window learner.

pdetransformer/core/pde_vittt_window.py
    Early window-local ViT3-style learner.

pdetransformer/core/mixed_channels/pde_transformer.py
    Token-mixer selection and local Attention--TTT integration.

server_example/train_ttt_ape_xxl_server.py
    Historical YAML-driven training entrypoint.
```

Temporal, rollout-TBPTT, and persistent-state experiments are intentionally
not included in this maintained screening branch. Their original experiment
branches remain available as archival Git references.
