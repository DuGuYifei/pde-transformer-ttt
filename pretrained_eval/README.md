# Pretrained-model evaluation

Evaluate an **off-the-shelf pretrained PDE-Transformer** on the *same* dataset
your server model is tested on, using the *same* nRMSE protocol — so you can put
the published model's numbers next to your own TTT-trained model's numbers.

By default it loads the official mixed-channel small model `thuerey-group/pde-transformer`
→ subfolder `mc-s` (`config.json` + `diffusion_pytorch_model.safetensors`, ~187 MB).

`test_pretrained_mc_server.py` is **self-contained**: it imports only the installed
`pdetransformer` package (exactly like `server_example/test_ttt_ape_xxl_server.py`),
so you can drop the single file into any run directory and run it — no sibling
`server_example/` folder needed.

## Why this is a fair, drop-in comparison

The official `mc-s` model is the **same architecture** your server model uses
(`in=2, out=2, sample_size=128, PDE-S, patch_size=4, periodic, no carrier token`),
the only difference being your model adds TTT window attention. `from_pretrained`
reads the published `config.json`, builds the plain (no-TTT) net, and loads the
weights with no missing/unexpected keys. Because there is no TTT state, this runs
**cache OFF only**.

The data pipeline params are identical to the server test (`downsample_factor=2`,
`max_channels=2`, `mean-std` norm), so both models see the same test inputs.

## ⚠️ Read before comparing numbers

The official model was trained at **full resolution** (`downsample_factor=1`) on
standard-resolution APEBench. Your server data is generated **low-res** and the
data module **downsamples by 2**. Keeping your pipeline means the pretrained model
is evaluated **zero-shot / out of distribution** on your low-res data — that's the
"does the off-the-shelf model already work on my data?" question, but it also means
these nRMSE values are **not** comparable to the paper's headline numbers
(paper MC-S: nRMSE1 ≈ 0.044, nRMSE10 ≈ 0.36 on full-res pretraining data).

---

## Running it on `cube4.pbs.cit.tum.de`

Confirmed server layout (from inspection):

| thing | location |
|---|---|
| data | `~/working/datasets/` (all `*.hdf5` present) |
| venv (py3.12, has `pdetransformer-ttt 0.0.1rc4`, `diffusers`, `huggingface_hub`) | `~/venv` |
| your low-res run root | `~/working/ttt_cache_low_res/` |
| this script (already copied) | `~/test/test_pretrained_mc_server.py` |

> ⚠️ Re-copy the script. The `~/test/` copy is the old import-based version. Push the
> current self-contained `test_pretrained_mc_server.py` to `~/test/` before running.

### Step 1 — pre-download the model on the **login node**

The HF cache does not exist yet (`~/.cache/huggingface` is absent), and PBS compute
nodes often have no internet. Download once on the login node (which does have
internet), into a folder under `~/working`:

```bash
source ~/venv/bin/activate
hf download thuerey-group/pde-transformer --include "mc-s/*" \
    --local-dir ~/working/pretrained_weights/pde-transformer
```

(`huggingface-cli` is deprecated/non-functional in this hub version — use `hf`,
same flags.) This creates:

```
~/working/pretrained_weights/pde-transformer/mc-s/config.json
~/working/pretrained_weights/pde-transformer/mc-s/diffusion_pytorch_model.safetensors
```

(Python fallback, if `hf` misbehaves:
`python -c "from huggingface_hub import snapshot_download; snapshot_download('thuerey-group/pde-transformer', allow_patterns='mc-s/*', local_dir='$HOME/working/pretrained_weights/pde-transformer')"`)

### Step 2 — run the evaluation

```bash
source ~/venv/bin/activate
cd ~/test
CUDA_VISIBLE_DEVICES=0 python test_pretrained_mc_server.py \
    --model-source ~/working/pretrained_weights/pde-transformer \
    --subfolder mc-s
```

Defaults already point at `~/working/datasets` for data and
`~/working/ttt_cache_low_res` as the output root — no `--config` needed.

**Quick smoke test first** (one PDE, 2 batches, ~a minute):

```bash
CUDA_VISIBLE_DEVICES=0 python test_pretrained_mc_server.py \
    --model-source ~/working/pretrained_weights/pde-transformer --subfolder mc-s \
    --datasets burgers --max-batches-per-dataset 2
```

If the compute node *does* have internet you can skip Step 1 and drop
`--model-source` entirely (it auto-downloads to the HF cache).

Other published sizes: `--subfolder mc-b` or `mc-l` (download those subfolders the
same way).

## Output

Written to `~/working/ttt_cache_low_res/pretrained_eval/mc-s/<timestamp>/`
(override with `--output-dir`):

- `results_cache_off.json` / `results_cache_off.csv` — per-dataset nRMSE_1/10/20 +
  `macro_avg` (mean over datasets) and `micro_avg` (mean over trajectories).
- `summary.json` — aggregates + run metadata.

The schema matches `server_example/test_ttt_ape_xxl_server.py`, so you can diff the
`macro_avg` / `micro_avg` rows directly against your own model's
`results_cache_off.csv` (your TTT model additionally has a `results_cache_on.csv`).

## Pipeline-parity note (verified)

The inlined `build_data_module` here matches `server_example/train_ttt_ape_xxl_server.py`.
Verified via git (`git log -S "downsample_factor"`): the `downsample_factor=2` /
`max_channels=2` data params were set in the **initial** server commit (`b28f399`)
and have **never changed since** — so this pipeline is identical to the one your
`server_01`/`server_02` test runs used. The only later edits to that script added
PDE-B `model_type` plumbing, which does not touch the data pipeline.
