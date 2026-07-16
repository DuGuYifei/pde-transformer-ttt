# Global ViTTT PDE experiment

This experiment keeps the original PDE-S U-Net, conditional AdaLN, residual
gates, MLPs, downsampling, upsampling, and skip connections. Only the token
mixer inside each PDE block is selectable.

| `token_mixer_type` | Mixer | Position encoding |
| --- | --- | --- |
| `attention` | Original shifted 8x8 attention | Original relative position bias |
| `global_vittt` | Full-map ViTTT + PDE MLP | Per-block 3x3 CPE |
| `global_h_vittt` | Full-map ViTTT + H-style Conv MLP | Per-block 3x3 CPE plus periodic 2D RoPE on Q/K |

Both ViTTT variants use the same `GlobalViTTTMixer` with `vittt_head_dim: 32`.
The H-style block additionally enables periodic 2D RoPE and the official
Conv-enhanced MLP structure; its depthwise convolution uses whole-domain
circular padding. The periodic RoPE is a toroidal PDE adaptation of the
official axial RoPE, not a byte-for-byte copy of that component. At 128x128
input resolution, the full-map token counts and TTT head layouts are:

```text
128x128 input
  -> patch embed: 32x32, N=1024, D=96,  heads=3
  -> encoder:     16x16, N=256,  D=192, heads=6
  -> latent:       8x8,  N=64,   D=384, heads=12
  -> decoder:     16x16, N=256,  D=192, heads=6
  -> decoder:     32x32, N=1024, D=192, heads=6
```

There is no window partition, shifted window, cache, or cross-PDE-step
fast-weight persistence in either ViTTT configuration. Fast weights start from
their learned initialization on every block call and are updated once over the
complete feature map.

Run the local checks before deployment:

```bash
python smoke_test/smoke_test_pde_global_vittt.py
```

Build a configured model without loading data:

```bash
python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-vittt_128_60sims.yaml --check-config
```

Run a two-GPU 20-epoch screening experiment on the server:

```bash
CUDA_VISIBLE_DEVICES=0,1 python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-vittt_128_60sims.yaml
```

Use `pdes_global-h-vittt_128_60sims.yaml` for the RoPE ablation or
`pdes_attention_128_60sims.yaml` for the matched baseline. Pass
`--max-epochs 100` only after the 20-epoch screening result is accepted.

For a selective server deployment, transfer the following source files plus
the selected YAML and training entrypoint:

```text
pdetransformer/core/pde_vittt_global.py
pdetransformer/core/mixed_channels/pde_transformer.py
server_example/train_global_vittt_ape_xxl_server.py
server_example/*.yaml
```

## Global Linear TTT with temporal adaptation

`train_global_linear_temporal_server.py` trains one persistent TTT-MLP after
the `8x8x384` latent stage of a pretrained Global Linear TTT backbone. The
matched configuration is
`pdes_global-linear-temporal-ttt-frozenbase_128_100ep_60sims.yaml`.

The 22 spatial Global Linear TTT blocks remain frozen and reset their fast
weights on every PDE model call. Only the latent temporal TTT is trainable;
its state persists for all 29 rollout steps. `tbptt_chunk_size: 4` detaches
the backward graph every four steps without resetting that state.

Set `temporal_ttt_enabled: false` for a pure G-L model. A model constructed
with the temporal module can also bypass it per call with
`use_temporal_ttt=False`, preserving the original G-L tensor path.
