#!/usr/bin/env python3
"""Summarize three model-seed Full-256 and generated ID/OOD evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


MODEL_SEEDS = (42, 43, 44)
STEPS = tuple(range(1, 30))
CONDITIONS = ("all", "id", "ood_low", "ood_high")
T_CRITICAL_95_DF2 = 4.302652729696142


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_root",
        type=Path,
        help="Directory containing seed42, seed43, and seed44 evaluation folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RESULT_ROOT/summary_3seed.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def metric_values(metrics: dict[str, Any]) -> dict[int, float]:
    return {step: float(metrics[f"nRMSE_{step}"]) for step in STEPS}


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stats(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if count > 1 else 0.0
    half_width = T_CRITICAL_95_DF2 * sample_std / math.sqrt(count) if count > 1 else 0.0
    return {
        "model_seed_count": count,
        "mean": mean,
        "sample_std": sample_std,
        "ci95_half_width": half_width,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def step_columns() -> list[str]:
    return [f"nRMSE_{step}" for step in STEPS]


def format_value(value: float) -> str:
    return f"{value:.6f}"


def render_report(
    output_dir: Path,
    aggregate_rows: list[dict[str, Any]],
    aggregate_stats_rows: list[dict[str, Any]],
    elapsed_rows: list[dict[str, Any]],
) -> None:
    per_seed = {
        (row["model_seed"], row["split"], row["condition"], row["aggregation"]): row
        for row in aggregate_rows
    }
    aggregate_stats = {
        (row["split"], row["condition"], row["aggregation"], row["step"]): row
        for row in aggregate_stats_rows
    }
    elapsed = {
        (row["model_seed"], row["split"]): float(row["elapsed_seconds"])
        for row in elapsed_rows
    }

    lines = [
        "# G-L+EMA PDE-S Full-256 Three-Seed Evaluation",
        "",
        "## Scope",
        "",
        "This report evaluates three independently trained G-L+EMA PDE-S models ",
        "with model-training seeds 42, 43, and 44. Every checkpoint uses its own ",
        "EMA-best model selection. These model seeds are distinct from the fixed ",
        "data-generation seeds inside the generated ID/OOD test matrix.",
        "",
        "- training resolution: 256 x 256",
        "- training data: `pde-transformer-ape2d-full` train split",
        "- training epochs: 100 per seed",
        "- inference checkpoint: `ema-best.ckpt`",
        "- evaluation batch: 8 on one GTX 1080 Ti",
        "- rollout: 29 autoregressive transitions",
        "- uncertainty: sample standard deviation and two-sided 95% Student-t CI ",
        "  over three independent model-training seeds (`df=2`)",
        "",
        "## Key Results By Model Seed",
        "",
        "### Full-256 Strict Test (Macro nRMSE)",
        "",
        "| Model seed | @1 | @10 | @20 | @29 | Time |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in MODEL_SEEDS:
        row = per_seed[(seed, "full_test", "all", "macro")]
        lines.append(
            "| %d | %s | %s | %s | %s | %.1f s |"
            % (
                seed,
                format_value(float(row["nRMSE_1"])),
                format_value(float(row["nRMSE_10"])),
                format_value(float(row["nRMSE_20"])),
                format_value(float(row["nRMSE_29"])),
                elapsed[(seed, "full_test")],
            )
        )

    lines.extend(
        [
            "",
            "### Generated ID/OOD Test, All Conditions (Macro nRMSE)",
            "",
            "| Model seed | @1 | @10 | @20 | @29 | Time |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in MODEL_SEEDS:
        row = per_seed[(seed, "id_ood", "all", "macro")]
        lines.append(
            "| %d | %s | %s | %s | %s | %.1f s |"
            % (
                seed,
                format_value(float(row["nRMSE_1"])),
                format_value(float(row["nRMSE_10"])),
                format_value(float(row["nRMSE_20"])),
                format_value(float(row["nRMSE_29"])),
                elapsed[(seed, "id_ood")],
            )
        )

    section_names = {
        ("full_test", "all"): "Full-256 Strict Test",
        ("id_ood", "all"): "Generated ID/OOD Test: All Conditions",
        ("id_ood", "id"): "Generated Test: ID",
        ("id_ood", "ood_low"): "Generated Test: OOD-Low",
        ("id_ood", "ood_high"): "Generated Test: OOD-High",
    }
    lines.extend(["", "## Complete 29-Step Aggregate Results", ""])
    for (split, condition), section_name in section_names.items():
        lines.extend(
            [
                f"### {section_name}",
                "",
                "| Step | Macro mean | Macro SD | Macro 95% CI | Micro mean | Micro SD | Micro 95% CI |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for step in STEPS:
            macro = aggregate_stats[(split, condition, "macro", step)]
            micro = aggregate_stats[(split, condition, "micro", step)]
            lines.append(
                "| %d | %s | %s | [%s, %s] | %s | %s | [%s, %s] |"
                % (
                    step,
                    format_value(float(macro["mean"])),
                    format_value(float(macro["sample_std"])),
                    format_value(float(macro["ci95_low"])),
                    format_value(float(macro["ci95_high"])),
                    format_value(float(micro["mean"])),
                    format_value(float(micro["sample_std"])),
                    format_value(float(micro["ci95_low"])),
                    format_value(float(micro["ci95_high"])),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- The confidence intervals quantify variation across three independently ",
            "  trained models. They are not trajectory-level bootstrap intervals.",
            "- With only three model seeds, the Student-t intervals are intentionally ",
            "  wide and should not be presented as strong significance evidence.",
            "- Macro gives every PDE equal weight. Micro weights PDEs by their number ",
            "  of evaluated trajectories.",
            "- Macro and micro coincide in the generated test because every ",
            "  PDE-condition cell contains the same number of trajectories.",
            "",
            "## Complete Artifacts",
            "",
            "- [Aggregate values for every model seed](aggregate_per_model_seed.csv)",
            "- [Aggregate mean, SD, and 95% CI for all 29 steps](aggregate_statistics.csv)",
            "- [Per-PDE values for every model seed and all 29 steps](per_pde_per_model_seed.csv)",
            "- [Per-PDE mean, SD, and 95% CI for all 29 steps](per_pde_statistics.csv)",
            "- [Evaluation times](evaluation_time.csv)",
            "- [Machine-readable complete summary](summary_3seed.json)",
            "",
            "Raw evaluator JSON, CSV, and logs are retained under `../seed42`, ",
            "`../seed43`, and `../seed44`.",
        ]
    )
    with (output_dir / "report.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_dir = (args.output_dir or result_root / "summary_3seed").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, Any]] = []
    per_pde_rows: list[dict[str, Any]] = []
    elapsed_rows: list[dict[str, Any]] = []

    for seed in MODEL_SEEDS:
        seed_root = result_root / f"seed{seed}"
        for split in ("full_test", "id_ood"):
            summary = load_json(seed_root / split / "summary.json")
            results = load_json(seed_root / split / "results_cache_off.json")
            aggregate = summary["aggregates_by_cache_mode"]["off"]
            elapsed_rows.append(
                {
                    "model_seed": seed,
                    "split": split,
                    "elapsed_seconds": summary["metadata"]["overall_elapsed_seconds"],
                }
            )

            conditions = ("all",) if split == "full_test" else CONDITIONS
            for condition in conditions:
                condition_aggregate = (
                    aggregate if condition == "all" else aggregate["conditions"][condition]
                )
                for aggregation in ("macro", "micro"):
                    values = metric_values(condition_aggregate[aggregation])
                    aggregate_rows.append(
                        {
                            "model_seed": seed,
                            "split": split,
                            "condition": condition,
                            "aggregation": aggregation,
                            **{f"nRMSE_{step}": values[step] for step in STEPS},
                        }
                    )

            for pde, pde_result in results["per_dataset"].items():
                pde_conditions = ("all",) if split == "full_test" else CONDITIONS
                for condition in pde_conditions:
                    source = pde_result if condition == "all" else pde_result["conditions"][condition]
                    values = metric_values(source)
                    per_pde_rows.append(
                        {
                            "model_seed": seed,
                            "split": split,
                            "condition": condition,
                            "pde": pde,
                            **{f"nRMSE_{step}": values[step] for step in STEPS},
                        }
                    )

    aggregate_stats_rows: list[dict[str, Any]] = []
    aggregate_keys = sorted(
        {(row["split"], row["condition"], row["aggregation"]) for row in aggregate_rows}
    )
    for split, condition, aggregation in aggregate_keys:
        selected = [
            row
            for row in aggregate_rows
            if (row["split"], row["condition"], row["aggregation"])
            == (split, condition, aggregation)
        ]
        for step in STEPS:
            aggregate_stats_rows.append(
                {
                    "split": split,
                    "condition": condition,
                    "aggregation": aggregation,
                    "step": step,
                    **stats([float(row[f"nRMSE_{step}"]) for row in selected]),
                }
            )

    per_pde_stats_rows: list[dict[str, Any]] = []
    pde_keys = sorted(
        {(row["split"], row["condition"], row["pde"]) for row in per_pde_rows}
    )
    for split, condition, pde in pde_keys:
        selected = [
            row
            for row in per_pde_rows
            if (row["split"], row["condition"], row["pde"])
            == (split, condition, pde)
        ]
        for step in STEPS:
            per_pde_stats_rows.append(
                {
                    "split": split,
                    "condition": condition,
                    "pde": pde,
                    "step": step,
                    **stats([float(row[f"nRMSE_{step}"]) for row in selected]),
                }
            )

    write_csv(
        output_dir / "aggregate_per_model_seed.csv",
        aggregate_rows,
        ["model_seed", "split", "condition", "aggregation", *step_columns()],
    )
    write_csv(
        output_dir / "aggregate_statistics.csv",
        aggregate_stats_rows,
        [
            "split",
            "condition",
            "aggregation",
            "step",
            "model_seed_count",
            "mean",
            "sample_std",
            "ci95_half_width",
            "ci95_low",
            "ci95_high",
        ],
    )
    write_csv(
        output_dir / "per_pde_per_model_seed.csv",
        per_pde_rows,
        ["model_seed", "split", "condition", "pde", *step_columns()],
    )
    write_csv(
        output_dir / "per_pde_statistics.csv",
        per_pde_stats_rows,
        [
            "split",
            "condition",
            "pde",
            "step",
            "model_seed_count",
            "mean",
            "sample_std",
            "ci95_half_width",
            "ci95_low",
            "ci95_high",
        ],
    )
    write_csv(
        output_dir / "evaluation_time.csv",
        elapsed_rows,
        ["model_seed", "split", "elapsed_seconds"],
    )

    serialized = {
        "model_seeds": MODEL_SEEDS,
        "steps": STEPS,
        "ci_method": "two-sided Student t, 95%, df=2",
        "aggregate_per_model_seed": aggregate_rows,
        "aggregate_statistics": aggregate_stats_rows,
        "per_pde_per_model_seed": per_pde_rows,
        "per_pde_statistics": per_pde_stats_rows,
        "evaluation_time": elapsed_rows,
    }
    with (output_dir / "summary_3seed.json").open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2)
        handle.write("\n")

    render_report(output_dir, aggregate_rows, aggregate_stats_rows, elapsed_rows)

    print(f"Wrote three-seed summary to {output_dir}")


if __name__ == "__main__":
    main()
