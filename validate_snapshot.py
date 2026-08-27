"""Validate the retained thesis evidence against the reported architecture screen."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


ARCHITECTURE_SCREEN = {
    "PDE-Transformer-S": (
        "architecture-screen-baseline/summary.json",
        (0.1065, 0.8114, 1.0208, 1.0957, 338.3, 33.19),
    ),
    "Window TTT-Linear": (
        "architecture-screen-window-ttt-linear/official_best_epoch096/summary.json",
        (0.1043, 0.8390, 1.0301, 1.1077, 313.2, 33.31),
    ),
    "Window TTT-MLP": (
        "architecture-screen-window-ttt-mlp/official_best_epoch099/summary.json",
        (0.1040, 0.8623, 0.9860, 1.0916, 406.3, 34.55),
    ),
    "Window Sequential TTT-Linear": (
        "architecture-screen-window-sequential/linear/official_best_epoch099/summary.json",
        (0.1062, 0.8285, 0.9651, 1.0433, 370.9, 33.31),
    ),
    "Window Sequential TTT-MLP": (
        "architecture-screen-window-sequential/mlp/official_best_epoch099/summary.json",
        (0.1092, 0.8485, 0.9980, 1.0569, 574.1, 34.55),
    ),
    "Window ViT3": (
        "architecture-screen-window-vit3/evaluation/summary.json",
        (0.1036, 0.8575, 0.9858, 1.0580, 429.0, 37.23),
    ),
    "Window Attention-TTT Hybrid": (
        "architecture-screen-window-attention-ttt-hybrid/evaluation/summary.json",
        (0.1065, 0.8305, 1.0623, 1.1749, 967.6, 43.12),
    ),
    "Full-map ViT3": (
        "architecture-screen-full-map-vit3/summary.json",
        (0.0902, 0.6384, 0.9635, 1.0723, 460.4, 36.13),
    ),
    "PDE-TTT-S": (
        "architecture-screen-pde-ttt/summary.json",
        (0.0939, 0.6387, 0.9443, 1.0355, 329.8, 33.36),
    ),
}


def rounded_summary(path: Path) -> tuple[float, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload["aggregates_by_cache_mode"]["off"]["macro"]
    metadata = payload["metadata"]
    params_m = float(str(payload["params"]["total_m"]).removesuffix("M"))
    return (
        round(float(aggregate["nRMSE_1"]), 4),
        round(float(aggregate["nRMSE_10"]), 4),
        round(float(aggregate["nRMSE_20"]), 4),
        round(float(aggregate["nRMSE_29"]), 4),
        round(float(metadata["overall_elapsed_seconds"]), 1),
        round(params_m, 2),
    )


def validate_architecture_screen() -> None:
    for label, (relative_path, expected) in ARCHITECTURE_SCREEN.items():
        actual = rounded_summary(ROOT / relative_path)
        if actual != expected:
            raise AssertionError(f"{label}: expected {expected}, got {actual}")


def validate_full256_evidence() -> None:
    required = [
        "global-linear-full256-evaluation/plain_attention/full_test/results_cache_off.csv",
        "global-linear-full256-evaluation/full_test/results_cache_off.csv",
        "global-linear-s-ema-3seed-full256-evaluation/summary_3seed/per_pde_per_model_seed.csv",
        "global-linear-s-rawbest-3seed-full256-evaluation/summary_3seed/per_pde_per_model_seed.csv",
        "global-linear-b-ema-full256-evaluation/ema/full_test/results_cache_off.csv",
        "efficiency-benchmarks/p-vs-global-linear-ttt/full256/attention_run1.json",
        "efficiency-benchmarks/p-vs-global-linear-ttt/full256/global_linear_ttt_run1.json",
        "figure-sources/data/full256_all29_family_statistics.csv",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Full-256 evidence: {missing}")


def validate_rollout_fields() -> None:
    root = ROOT / "figure-sources/autoregressive-predictions/rollout_fields"
    expected_models = (
        "pde_transformer_s",
        "pde_ttt_s",
        "pde_ttt_s_ema",
        "published_pde_s",
    )
    for model in expected_models:
        files = list((root / model).glob("*.npz"))
        if len(files) != 16:
            raise AssertionError(f"{model}: expected 16 NPZ files, got {len(files)}")
        if not (root / model / "manifest.json").is_file():
            raise FileNotFoundError(f"Missing rollout manifest for {model}")


if __name__ == "__main__":
    validate_architecture_screen()
    validate_full256_evidence()
    validate_rollout_fields()
    print("Thesis evidence snapshot validation passed.")

