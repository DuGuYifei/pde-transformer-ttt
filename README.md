# Full-map ViT3 for PDE-Transformer

This branch contains the full-map ViT3 architecture used in the thesis
architecture screen. It adapts the H-style design from Vision Test-Time
Training to the PDE-Transformer backbone:

- full-map depthwise convolutional positional encoding;
- periodic 2D RoPE for periodic PDE fields;
- nonlinear SwiGLU and dynamic depthwise-convolution fast-weight branches;
- a convolution-enhanced outer MLP; and
- one independent temporary fast-weight update per block and model call.

The implementation is an architecture-screen branch. The final linear
PDE-TTT model is maintained on
[`thesis/pde-ttt`](https://github.com/DuGuYifei/pde-transformer-ttt/tree/thesis/pde-ttt).
The base architecture comes from
[PDE-Transformer](https://github.com/tum-pbs/pde-transformer).

## Install

```bash
git clone --branch thesis/full-map-vit3 \
  https://github.com/DuGuYifei/pde-transformer-ttt.git
cd pde-transformer-ttt
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Train the recorded 128-resolution screen

```bash
python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-h-vittt_128_60sims.yaml \
  --check-config

python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_global-h-vittt_128_60sims.yaml
```

The YAML expects the thesis `datasets` directory used by the 128-resolution
architecture screen. Data paths and hardware settings may be overridden with
`--run-root`, `--devices`, `--batch-size`, and
`--accumulate-grad-batches`.

## Evaluate

```bash
python pretrained_eval/test_pretrained_mc_server.py \
  --config server_example/pdes_global-h-vittt_128_60sims.yaml \
  --checkpoint-path /path/to/checkpoint.ckpt \
  --data-dir /path/to/evaluation/data \
  --batch-size 8 \
  --output-dir /path/to/results
```

## Verify

```bash
python smoke_test/smoke_test_pde_global_vittt.py
```

See [`LICENSE.txt`](LICENSE.txt) for licensing information.
