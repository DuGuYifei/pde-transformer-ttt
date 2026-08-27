# P vs Global Linear TTT Efficiency Benchmark

## Purpose

This benchmark compares the model-level training cost of the matched 128-resolution
PDE-S models:

- **P**: original `8x8` shifted-window Attention.
- **G-L**: full-map Global Linear TTT replacing Attention.

It measures complete forward, MSE backward, and AdamW update work. It does not
measure prediction quality; official nRMSE evaluation remains a separate experiment.

## Environment and protocol

| Setting | Value |
|---|---|
| Date | 2026-07-16 |
| Server | `cube4` |
| GPU | NVIDIA GeForce GTX 1080 Ti, one physical GPU (`CUDA_VISIBLE_DEVICES=2`) |
| PyTorch | `2.5.1+cu121` |
| Precision | FP32 |
| Model | PDE-S, periodic, patch size 4, input/output channels 2 |
| Input | Synthetic `[8, 2, 128, 128]` tensor |
| Per-GPU batch | 8 |
| Gradient accumulation | 8 batches |
| Optimizer | AdamW, learning rate `4e-5`, weight decay `1e-15` |
| Warm-up | 10 training steps |
| Measurement | 50 training steps per process |
| Repetitions | Two independent processes per model |
| Execution order | P run 1, G-L run 1, G-L run 2, P run 2 |
| Seed | 42 |

Each process loaded the reviewed source from
`server_09_global_linear_ttt/pdetransformer`, not the older installed package.
Peak memory is reported by PyTorch's CUDA allocator after the optimizer state has
been materialized during warm-up.

## Parameters

| Model | Parameters | FP32 parameter memory | Difference from P |
|---|---:|---:|---:|
| P | 33,190,328 | 126.611 MiB | baseline |
| G-L | 33,361,952 | 127.266 MiB | +171,624 parameters (+0.517%); +0.655 MiB |

G-L therefore has slightly more persistent parameter storage. Any total-memory
reduction comes from lower training activation/intermediate cost, not from having
fewer parameters.

## Raw runs

| Model/run | Mean step | Median | P95 | Throughput | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| P run 1 | 116.382 ms | 114.691 ms | 123.795 ms | 68.739 samples/s | 2772.908 MiB | 2846 MiB |
| P run 2 | 116.680 ms | 116.129 ms | 123.303 ms | 68.564 samples/s | 2772.908 MiB | 2858 MiB |
| G-L run 1 | 102.086 ms | 101.134 ms | 110.361 ms | 78.365 samples/s | 2510.571 MiB | 2590 MiB |
| G-L run 2 | 104.473 ms | 103.624 ms | 112.234 ms | 76.575 samples/s | 2510.571 MiB | 2590 MiB |

## Averaged comparison

| Metric | P | G-L | G-L relative to P |
|---|---:|---:|---:|
| Mean training-step time | 116.531 ms | 103.280 ms | **11.372% lower; 1.128x speedup** |
| Throughput | 68.651 samples/s | 77.470 samples/s | **12.846% higher** |
| Peak allocated memory | 2772.908 MiB | 2510.571 MiB | **262.337 MiB / 9.461% lower** |
| Peak reserved memory | 2852 MiB | 2590 MiB | **262 MiB / 9.187% lower** |

Under this controlled model-level workload, G-L is both faster and less
activation-memory intensive than P despite its 0.517% parameter increase.

The complete matched training logs independently show `1.9 min/epoch` for P and
`1.6 min/epoch` for G-L at epochs 1, 50, and 99. Those end-to-end values include
additional training-system effects and are corroborating evidence, not inputs to
the microbenchmark averages above.

## Scope and limitations

- This is a single-GPU model microbenchmark. It excludes dataloader work, DDP/NCCL,
  validation, checkpoint I/O, and dataset normalization.
- Synthetic tensors isolate architecture cost but do not reproduce HDF5 input
  latency or PDE-dependent sample distributions.
- Results apply to PDE-S at resolution 128, FP32, batch 8 on a GTX 1080 Ti. They
  should not be generalized to other GPUs, precision modes, resolutions, or batch
  sizes without a new benchmark.
- Training-step results do not establish inference latency. A separate forward-only
  benchmark is required for that claim.
- Efficiency does not establish model quality. G-L's official `@1/@10/@20/@29`
  nRMSE must be reported separately.

## Reproduction artifacts

- Benchmark script:
  `pde-transformer-ttt/benchmark/benchmark_p_vs_global_linear_ttt.py`
- Source branch and implementation commit:
  `experiment/global-ttt-linear-v1`, `0ad3f06`
- Raw measurements:
  `attention_run1.json`, `attention_run2.json`,
  `global_linear_ttt_run1.json`, `global_linear_ttt_run2.json`
