"""Smoke checks for the explicit full APEBench simulation split."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
for package_name, package_path in {
    "pdetransformer": REPO_ROOT / "pdetransformer",
    "pdetransformer.data": REPO_ROOT / "pdetransformer" / "data",
    "pdetransformer.data.pbdl_datatypes": (
        REPO_ROOT / "pdetransformer" / "data" / "pbdl_datatypes"
    ),
}.items():
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

from pdetransformer.data.pbdl_datatypes import ape_2d_xxl
from pdetransformer.data.pbdl_datatypes.ape_2d_splits import (
    PAPER_DATASET_NAMES,
    SEPARATE_TEST_DATASETS,
    ape_2d_xxl_simulation_split,
)


class RecordingDataset:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)

    def __len__(self):
        return 100


class RecordingSubset:
    def __init__(self):
        self.indices = [1, 0]


def main() -> None:
    assert len(PAPER_DATASET_NAMES) == 16
    assert "hyp" not in PAPER_DATASET_NAMES

    for name in PAPER_DATASET_NAMES:
        train_sims, test_sims = ape_2d_xxl_simulation_split(name, "full_paper")
        if name in SEPARATE_TEST_DATASETS:
            assert train_sims is None and test_sims is None
        else:
            assert train_sims
            assert test_sims
            assert set(train_sims).isdisjoint(test_sims)

    RecordingDataset.calls.clear()
    train_subset = RecordingSubset()
    val_subset = RecordingSubset()
    with (
        patch.object(ape_2d_xxl, "PBDLDataset", RecordingDataset),
        patch.object(
            ape_2d_xxl,
            "random_split",
            return_value=(train_subset, val_subset),
        ),
    ):
        train, val, test = ape_2d_xxl.ape_2d_xxl_datasets(
            dataset_name="diff",
            dataset_directory="/unused",
            unrolling_steps=1,
            test_unrolling_steps=29,
            dataset_profile="full_paper",
        )

    assert (train, val) == (train_subset, val_subset)
    assert isinstance(test, RecordingDataset)
    assert RecordingDataset.calls[0]["sel_sims"] == list(range(500))
    assert RecordingDataset.calls[1]["sel_sims"] == list(range(500, 600))
    assert set(RecordingDataset.calls[0]["sel_sims"]).isdisjoint(
        RecordingDataset.calls[1]["sel_sims"]
    )

    print("APE2D full split smoke test passed.")


if __name__ == "__main__":
    main()
