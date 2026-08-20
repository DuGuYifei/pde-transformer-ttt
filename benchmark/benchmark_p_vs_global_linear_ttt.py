from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F


MIXERS = ("attention", "global_linear_ttt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixer", choices=MIXERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def training_iteration(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    iteration: int,
    accumulation: int,
) -> None:
    if iteration % accumulation == 0:
        optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs, class_labels=labels).sample
    loss = F.mse_loss(predictions, targets) / accumulation
    loss.backward()
    if (iteration + 1) % accumulation == 0:
        optimizer.step()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))
    for package_name, package_path in {
        "pdetransformer": repository_root / "pdetransformer",
        "pdetransformer.core": repository_root / "pdetransformer" / "core",
    }.items():
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package
    from pdetransformer.core.mixed_channels.pde_transformer import PDETransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.warmup_steps < args.accumulate_grad_batches:
        raise ValueError("warmup-steps must include at least one optimizer update")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    model = PDETransformer(
        sample_size=args.sample_size,
        in_channels=2,
        out_channels=2,
        type="PDE-S",
        patch_size=4,
        periodic=True,
        carrier_token_active=False,
        token_mixer_type=args.mixer,
        vittt_inner_lr=1.0,
        vittt_head_dim=32,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.0e-5, weight_decay=1.0e-15)

    inputs = torch.randn(
        args.batch_size,
        2,
        args.sample_size,
        args.sample_size,
        device=device,
    )
    targets = torch.randn_like(inputs)
    labels = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(args.warmup_steps):
        training_iteration(
            model,
            optimizer,
            inputs,
            targets,
            labels,
            iteration,
            args.accumulate_grad_batches,
        )
    torch.cuda.synchronize(device)

    durations_ms: list[float] = []
    measured_start = time.perf_counter()
    for offset in range(args.measure_steps):
        iteration = args.warmup_steps + offset
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        training_iteration(
            model,
            optimizer,
            inputs,
            targets,
            labels,
            iteration,
            args.accumulate_grad_batches,
        )
        end.record()
        end.synchronize()
        durations_ms.append(start.elapsed_time(end))
    torch.cuda.synchronize(device)
    measured_wall_seconds = time.perf_counter() - measured_start

    result = {
        "mixer": args.mixer,
        "implementation_file": inspect.getfile(PDETransformer),
        "device": properties.name,
        "torch_version": torch.__version__,
        "precision": "fp32",
        "sample_size": args.sample_size,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "seed": args.seed,
        "parameters": parameter_count,
        "parameter_memory_mib_fp32": parameter_count * 4 / 2**20,
        "step_time_ms_mean": statistics.fmean(durations_ms),
        "step_time_ms_median": statistics.median(durations_ms),
        "step_time_ms_stdev": statistics.stdev(durations_ms),
        "step_time_ms_p95": percentile(durations_ms, 0.95),
        "samples_per_second": args.batch_size / (statistics.fmean(durations_ms) / 1000.0),
        "measured_wall_seconds": measured_wall_seconds,
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
