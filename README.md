# Window Attention-TTT hybrid experiment

This branch contains the Attention-TTT hybrid evaluated during the PDE-TTT
architecture screen. It extends the
[PDE-Transformer](https://github.com/tum-pbs/pde-transformer) window block
without replacing self-attention:

1. local W-MSA or SW-MSA is applied inside each 8 x 8 window;
2. the attention output passes through a forward-order TTT-MLP;
3. a second TTT-MLP pass uses the reversed token order;
4. learned gates add both temporary TTT outputs before the original
   PDE-conditioned residual and outer MLP.

The two TTT passes reuse the learned TTT-MLP initialization but reconstruct
temporary fast weights on every call. No fast-weight state is carried between
PDE rollout steps.

The final full-map linear model is maintained on
[`thesis/pde-ttt`](https://github.com/DuGuYifei/pde-transformer-ttt/tree/thesis/pde-ttt).

## Installation

```bash
git clone --branch thesis/window-attention-ttt-hybrid \
  https://github.com/DuGuYifei/pde-transformer-ttt.git
cd pde-transformer-ttt
python -m venv venv
source venv/bin/activate
pip install -e .
```

On PowerShell use `./venv/Scripts/Activate.ps1`.

## Train

The retained thesis configuration is
`server_example/pdes_attention-ttt-mlp-bidir_128_60sims.yaml`.

```bash
python server_example/train_ttt_ape_xxl_server.py \
  --config server_example/pdes_attention-ttt-mlp-bidir_128_60sims.yaml
```

## Evaluate

```bash
python pretrained_eval/test_pretrained_mc_server.py \
  --config server_example/pdes_attention-ttt-mlp-bidir_128_60sims.yaml \
  --checkpoint-path /path/to/checkpoint.ckpt \
  --data-dir ~/working/datasets \
  --batch-size 8 \
  --output-dir ~/working/eval/window-attention-ttt-hybrid
```

## Verify

```bash
python smoke_test/smoke_test_pde_attention_ttt.py
```

See `LICENSE.txt` and the upstream repository for license and citation
information.
