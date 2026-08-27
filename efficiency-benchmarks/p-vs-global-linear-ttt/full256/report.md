# PDE-Transformer-S vs PDE-TTT-S Full-256 Efficiency Benchmark

## Protocol

This controlled benchmark extends the existing 128-resolution measurement to
256-resolution synthetic inputs. It measures model computation and CUDA allocator
memory, not dataloader, DDP, validation, or checkpoint overhead.

| Setting | Value |
|---|---|
| Date | 2026-08-19 |
| Server | `cube4` |
| GPU | NVIDIA GeForce GTX 1080 Ti, one physical GPU (`CUDA_VISIBLE_DEVICES=2`) |
| PyTorch | `2.5.1+cu121` |
| Precision | FP32 |
| Model | PDE-S, periodic, patch size 4, input/output channels 2 |
| Input | Synthetic `[8, 2, 256, 256]` tensor |
| Per-GPU batch | 8 |
| Gradient accumulation | 8 microbatches |
| Optimizer | AdamW, learning rate `4e-5`, weight decay `1e-15` |
| Warm-up | 10 microbatches |
| Measurement | 50 microbatches per process |
| Repetitions | Two independent processes per model |
| Execution order | PDE-Transformer run 1, PDE-TTT run 1, PDE-TTT run 2, PDE-Transformer run 2 |
| Seed | 42 |
| Source | `experiment/full256-efficiency-benchmark-v1` at `6171127` |

Each measured microbatch includes forward propagation and MSE backward propagation.
Every eighth microbatch also includes an AdamW update. CUDA events measure elapsed
GPU time after warm-up.

## Raw runs

| Model/run | Mean microbatch | Median | P95 | Throughput | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| PDE-Transformer-S run 1 | 329.747 ms | 329.047 ms | 337.083 ms | 24.261 samples/s | 10483.439 MiB | 10726 MiB |
| PDE-Transformer-S run 2 | 330.201 ms | 329.399 ms | 337.066 ms | 24.228 samples/s | 10483.439 MiB | 10726 MiB |
| PDE-TTT-S run 1 | 310.151 ms | 309.247 ms | 317.364 ms | 25.794 samples/s | 9434.329 MiB | 9628 MiB |
| PDE-TTT-S run 2 | 309.236 ms | 308.487 ms | 316.244 ms | 25.870 samples/s | 9434.329 MiB | 9628 MiB |

The mean-time spread between repetitions is 0.138% for PDE-Transformer-S and
0.296% for PDE-TTT-S.

## Averaged comparison

| Metric | PDE-Transformer-S | PDE-TTT-S | Difference |
|---|---:|---:|---:|
| Mean microbatch time | 329.974 ms | **309.694 ms** | **6.15% lower** |
| Throughput | 24.244 samples/s | **25.832 samples/s** | **6.55% higher** |
| Peak allocated memory | 10483.439 MiB | **9434.329 MiB** | **1049.110 MiB / 10.01% lower** |
| Peak reserved memory | 10726 MiB | **9628 MiB** | **1098 MiB / 10.24% lower** |
| Parameters | **33.19M** | 33.36M | 0.52% higher |

## Scope

The measurements are stable across two independently launched processes. They
support a controlled 256-resolution comparison of model compute and allocator
memory on this GPU. They do not replace the separate end-to-end epoch-time logs,
which include real data loading, DDP communication, validation, and callbacks.

The benchmark implementation is fixed at the source commit above. The four JSON
files in this directory are the retained raw measurements.
