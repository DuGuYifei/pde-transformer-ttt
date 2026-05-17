# Server Example

This folder contains two small server entrypoints for continuing the Kaggle/notebook TTT experiment on a Linux server.

## Recommended layout

Copy the Kaggle working folder structure to the server as:

```bash
~/working/datasets
~/working/ttt_cache_experiments/train_once/checkpoints/last.ckpt
```

The training script defaults to that layout. If you put files somewhere else, edit `train_ttt_ape_xxl_server.yaml` or pass CLI overrides.

## Install

If the server does not have `git`, use the packaged PyPI release and copy only this `server_example` folder to the server:

```bash
pip install pdetransformer-ttt==0.0.1rc4
```

Then run the scripts from the copied folder. This works because the scripts import the installed `pdetransformer` package.

For continuing training from existing `~/working/datasets`, you do not need JAX or Exponax. Install them only if you want this server script to generate simulation data:

```bash
pip install "jax[cuda12]" exponax==0.1.0 h5py matplotlib vape4d
```

During active development, `git clone && pip install -e .` is still the best option when available, because it guarantees the scripts and package code come from the same commit:

```bash
git clone <your-repo-url> pde-transformer-ttt
cd pde-transformer-ttt
pip install -e .
```

## 1. GPU smoke test

Use `CUDA_VISIBLE_DEVICES` to decide which GPUs are visible. The smoke script uses all visible GPUs by default, or a fixed number with `--devices`.

```bash
CUDA_VISIBLE_DEVICES=0 python server_example/smoke_2gpu.py --devices 1
CUDA_VISIBLE_DEVICES=0,1 python server_example/smoke_2gpu.py
CUDA_VISIBLE_DEVICES=0,1,2 python server_example/smoke_2gpu.py --devices 3
```

On the current server, GPU 2 showed `ERR!` in `nvidia-smi`, so prefer GPU 0 and 1 first:

```bash
CUDA_VISIBLE_DEVICES=0,1 python server_example/smoke_2gpu.py
```

Passing the smoke test means PyTorch, Lightning, CUDA, and the selected distributed strategy can run a tiny training loop. It does not prove the full PDE/TTT training will fit memory or preserve the exact single-GPU training semantics.

## 2. Continue training

Safest first resumed run, matching the notebook/Kaggle setup as closely as possible:

```bash
CUDA_VISIBLE_DEVICES=0 python server_example/train_ttt_ape_xxl_server.py --config server_example/train_ttt_ape_xxl_server.yaml
```

The default YAML uses:

```yaml
devices: 1
strategy: auto
precision: 32-true
accumulate_grad_batches: 8
auto_resume: true
```

It will auto-resume from the newest `last*.ckpt` in:

```bash
~/working/ttt_cache_experiments/train_once/checkpoints/
```

For example, `last-v4.ckpt` is preferred over an older `last.ckpt`.

## 3. Optional 2-GPU training

After the smoke test passes, you can try:

```bash
CUDA_VISIBLE_DEVICES=0,1 python server_example/train_ttt_ape_xxl_server.py \
  --config server_example/train_ttt_ape_xxl_server.yaml \
  --devices 2 \
  --strategy ddp \
  --accumulate-grad-batches 4
```

Multi-GPU changes the effective global batch size because `batch_size` is per process/device in Lightning DDP. To stay close to the original single-GPU Kaggle run with `accumulate_grad_batches: 8`, use `accumulate_grad_batches: 4` for 2 GPUs.

## 4. Optional data generation

Data generation is off by default. `resume` and `auto_resume` only control checkpoints; they do not generate data. To generate the full APE XXL set before training, set this in `train_ttt_ape_xxl_server.yaml`:

```yaml
generate_data: true
low_res: true
force_regenerate_data: false
```

With `force_regenerate_data: false`, existing `*.hdf5` files are skipped so copied Kaggle data is not overwritten. Use `force_regenerate_data: true` only when you intentionally want to regenerate matching files.

You can also enable generation from the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python server_example/train_ttt_ape_xxl_server.py \
  --config server_example/train_ttt_ape_xxl_server.yaml \
  --generate-data
```

## 5. Useful overrides


Skip the rollout cache-off/cache-on evaluation after training:

```bash
python server_example/train_ttt_ape_xxl_server.py --config server_example/train_ttt_ape_xxl_server.yaml --skip-rollout-eval
```

Force a specific checkpoint:

```bash
python server_example/train_ttt_ape_xxl_server.py \
  --config server_example/train_ttt_ape_xxl_server.yaml \
  --resume \
  --checkpoint-path ~/working/ttt_cache_experiments/train_once/checkpoints/epoch-epoch=026.ckpt
```

Disable auto resume and start fresh:

```bash
python server_example/train_ttt_ape_xxl_server.py --config server_example/train_ttt_ape_xxl_server.yaml --no-auto-resume
```
