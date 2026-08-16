"""Explicit simulation splits for the extended 2D APE datasets."""

from __future__ import annotations


PAPER_DATASET_NAMES = (
    "diff",
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

LEGACY_DATASET_NAMES = (
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

SEPARATE_TEST_DATASETS = frozenset(
    {
        "ks",
        "gs_alpha",
        "gs_beta",
        "gs_gamma",
        "gs_epsilon",
        "decay_turb",
        "kolm_flow",
    }
)

JOINT_GS_DATASETS = frozenset({"gs_delta", "gs_theta", "gs_iota", "gs_kappa"})
JOINT_STANDARD_DATASETS = frozenset(
    {"adv", "diff", "adv_diff", "disp", "hyp", "burgers", "kdv", "fisher", "sh"}
)
DATASET_PROFILES = frozenset({"legacy_small", "full_paper"})


def dataset_names_for_profile(dataset_profile: str) -> tuple[str, ...]:
    if dataset_profile == "legacy_small":
        return LEGACY_DATASET_NAMES
    if dataset_profile == "full_paper":
        return PAPER_DATASET_NAMES
    raise ValueError(
        f"Unknown dataset_profile={dataset_profile!r}; expected one of "
        f"{sorted(DATASET_PROFILES)}."
    )


def ape_2d_xxl_simulation_split(
    dataset_name: str,
    dataset_profile: str,
) -> tuple[list[int] | None, list[int] | None]:
    """Return train-source and test simulation IDs for a dataset.

    A ``None`` pair denotes datasets whose test trajectories live in a
    separate ``*_test.hdf5`` file.
    """

    if dataset_profile not in DATASET_PROFILES:
        raise ValueError(
            f"Unknown dataset_profile={dataset_profile!r}; expected one of "
            f"{sorted(DATASET_PROFILES)}."
        )
    if dataset_name in SEPARATE_TEST_DATASETS:
        return None, None
    if dataset_name in JOINT_GS_DATASETS:
        return list(range(0, 80)), list(range(80, 100))
    if dataset_name in JOINT_STANDARD_DATASETS:
        split = 50 if dataset_profile == "legacy_small" else 500
        end = 60 if dataset_profile == "legacy_small" else 600
        return list(range(0, split)), list(range(split, end))
    raise ValueError(f"Unknown dataset: {dataset_name}")
