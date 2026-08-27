# Window TTT architecture experiments

This branch contains the window-local architecture screen for
[PDE-TTT](https://github.com/DuGuYifei/pde-transformer-ttt/tree/thesis/pde-ttt).
It is based on the original
[PDE-Transformer](https://github.com/tum-pbs/pde-transformer) backbone and keeps
the 8 x 8 shifted-window partition. Self-attention is replaced inside each
window by one of the TTT mixers below.

The final full-map linear PDE-TTT model is maintained on
`thesis/pde-ttt`; the Attention-TTT hybrid is maintained separately on
`thesis/window-attention-ttt-hybrid`.

## Included mixers

| Mixer | YAML | Update |
|---|---|---|
| Window ViT3 | `pdes_vittt-cacheoff_128_60sims.yaml` | ViT3-style two-branch window mixer |
| Window Full-Batch TTT-Linear | `pdes_window-linear-ttt_128_60sims.yaml` | One closed-form update per window |
| Window Full-Batch TTT-MLP | `pdes_window-fullbatch-mlp-ttt_128_60sims.yaml` | One non-causal MLP update per window |
| Window Token-Sequential TTT-Linear/MLP | `pdes_window-*-token-sequential_128_60sims.yaml` | Four groups of 16 tokens inside each window |

Fast weights are temporary and are reconstructed on every model call. None of
these configurations carries fast weights across PDE rollout steps.

## Installation

```bash
git clone --branch thesis/window-ttt \
  https://github.com/DuGuYifei/pde-transformer-ttt.git
cd pde-transformer-ttt
python -m venv venv
source venv/bin/activate
pip install -e .
```

On PowerShell use `./venv/Scripts/Activate.ps1`.

## Data

The 128-resolution architecture screen uses the small datasets generated with
the bundled APEBench simulation module. The final 256-resolution dataset is
available at
[`thuerey-group/pde-transformer-ape2d-full`](https://huggingface.co/datasets/thuerey-group/pde-transformer-ape2d-full).

To reproduce the thesis ID/OOD generation protocol at 256 resolution:

```bash
python -m pdetransformer.data.simulations_apebench.generate_id_ood_testset \
  --output-dir ~/working/datasets_test_256 \
  --gpu-id 0
```

## Train and evaluate

Choose one YAML from `server_example/`:

```bash
python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_window-linear-ttt_128_60sims.yaml \
  --check-config

python server_example/train_global_vittt_ape_xxl_server.py \
  --config server_example/pdes_window-linear-ttt_128_60sims.yaml
```

Evaluate a retained checkpoint with the same YAML:

```bash
python pretrained_eval/test_pretrained_mc_server.py \
  --config server_example/pdes_window-linear-ttt_128_60sims.yaml \
  --checkpoint-path /path/to/checkpoint.ckpt \
  --data-dir ~/working/datasets \
  --batch-size 8 \
  --output-dir ~/working/eval/window-linear
```

## Verification

```bash
python smoke_test/smoke_test_pde_vittt_window.py
python smoke_test/smoke_test_pde_window_linear_ttt.py
python smoke_test/smoke_test_pde_window_fullbatch_mlp_ttt.py
python smoke_test/smoke_test_pde_window_ttt_sequential.py
```

See `LICENSE.txt` and the upstream repository for license and citation
information.
