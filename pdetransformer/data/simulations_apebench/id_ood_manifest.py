"""Deterministic ID/OOD parameter matrix for the APEBench 2D test set."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


PDE_NAMES = (
    "diff",
    "hyp",
    "burgers",
    "kdv",
    "ks",
    "fisher",
    "gs_alpha",
    "gs_beta",
    "gs_gamma",
    "gs_delta",
    "gs_epsilon",
    "gs_theta",
    "gs_iota",
    "gs_kappa",
    "sh",
    "decay_turb",
    "kolm_flow",
)

CONDITIONS = ("id", "ood_low", "ood_high")
SEEDS = (2026072201, 2026072202, 2026072203)


def _scalar_conditions(
    key: str,
    train_min: float,
    train_max: float,
    *,
    fixed: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    fixed = {} if fixed is None else dict(fixed)
    center = (train_min + train_max) / 2.0
    return {
        "id": {**fixed, key: center},
        "ood_low": {**fixed, key: 0.95 * train_min},
        "ood_high": {**fixed, key: 1.05 * train_max},
    }


def _gray_scott_conditions(feed: float, kill: float) -> dict[str, dict[str, float]]:
    return {
        "id": {"Feed Rate": feed, "Kill Rate": kill},
        "ood_low": {"Feed Rate": 0.95 * feed, "Kill Rate": kill},
        "ood_high": {"Feed Rate": 1.05 * feed, "Kill Rate": kill},
    }


PARAMETER_MATRIX: dict[str, dict[str, dict[str, float]]] = {
    "diff": {
        "id": {"Viscosity X": 0.0275, "Viscosity Y": 0.0275},
        "ood_low": {"Viscosity X": 0.00475, "Viscosity Y": 0.00475},
        "ood_high": {"Viscosity X": 0.0525, "Viscosity Y": 0.0525},
    },
    "hyp": _scalar_conditions("Hyper-Diffusivity", 5.0e-5, 5.0e-4),
    "burgers": _scalar_conditions("Viscosity", 5.0e-5, 3.0e-4),
    "kdv": _scalar_conditions(
        "Viscosity",
        5.0e-5,
        1.0e-3,
        fixed={"Domain Extent": 75.0},
    ),
    "ks": _scalar_conditions("Domain Extent", 10.0, 130.0),
    "fisher": _scalar_conditions(
        "Diffusivity",
        1.0e-4,
        2.0e-2,
        fixed={"Reactivity": 10.0},
    ),
    "gs_alpha": _gray_scott_conditions(0.008, 0.046),
    "gs_beta": _gray_scott_conditions(0.020, 0.046),
    "gs_gamma": _gray_scott_conditions(0.024, 0.056),
    "gs_delta": _gray_scott_conditions(0.028, 0.056),
    "gs_epsilon": _gray_scott_conditions(0.020, 0.056),
    "gs_theta": _gray_scott_conditions(0.040, 0.060),
    "gs_iota": _gray_scott_conditions(0.050, 0.0605),
    "gs_kappa": _gray_scott_conditions(0.052, 0.063),
    "sh": _scalar_conditions(
        "Reactivity",
        0.4,
        1.0,
        fixed={"Critical Number": 1.0},
    ),
    "decay_turb": _scalar_conditions("Viscosity", 1.0e-5, 1.0e-4),
    "kolm_flow": _scalar_conditions("Viscosity", 1.0e-4, 1.0e-3),
}


def simulation_entries(pde: str) -> list[dict[str, Any]]:
    if pde not in PDE_NAMES:
        raise ValueError(f"Unknown PDE {pde!r}")
    entries = []
    sim_id = 0
    for condition in CONDITIONS:
        for seed in SEEDS:
            entries.append(
                {
                    "sim_id": sim_id,
                    "condition": condition,
                    "seed": seed,
                    "parameter_overrides": deepcopy(PARAMETER_MATRIX[pde][condition]),
                }
            )
            sim_id += 1
    return entries


def build_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "P/G-L 128-resolution ID and parameter-OOD evaluation",
        "solver_resolution": 2048,
        "stored_resolution": 256,
        "evaluation_resolution": 128,
        "downsampling": {
            "solver_to_storage": "non-overlapping 8x8 arithmetic mean",
            "storage_to_evaluation": "PyTorch avg_pool2d factor 2",
        },
        "time_steps": 30,
        "rollout_transitions": 29,
        "conditions": list(CONDITIONS),
        "ood_definition": "near-OOD: 5% beyond the varied training boundary",
        "seeds": list(SEEDS),
        "pdes": {pde: simulation_entries(pde) for pde in PDE_NAMES},
    }
