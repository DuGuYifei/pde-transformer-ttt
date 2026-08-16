from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
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
    SEPARATE_TEST_DATASETS,
    ape_2d_xxl_simulation_split,
)


OFFICIAL_DATASETS = [
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
]

SIMULATION_COUNTS = {name: 60 for name in OFFICIAL_DATASETS}
SIMULATION_COUNTS.update(
    {
        "gs_alpha": 10,
        "gs_beta": 10,
        "gs_gamma": 10,
        "gs_epsilon": 10,
        "gs_delta": 100,
        "gs_theta": 100,
        "gs_iota": 100,
        "gs_kappa": 100,
        "gs_alpha_test": 3,
        "gs_beta_test": 3,
        "gs_gamma_test": 3,
        "gs_epsilon_test": 3,
        "ks_test": 5,
        "decay_turb_test": 5,
        "kolm_flow_test": 5,
    }
)

FRAME_COUNTS = {name: 30 for name in OFFICIAL_DATASETS}
FRAME_COUNTS.update(
    {
        "gs_alpha_test": 100,
        "gs_beta_test": 100,
        "gs_gamma_test": 100,
        "gs_epsilon_test": 100,
        "ks_test": 200,
        "decay_turb_test": 200,
        "kolm_flow_test": 200,
    }
)


@dataclass
class FakePBDLDataset:
    dset_name: str
    local_datasets_dir: str
    time_steps: int
    intermediate_time_steps: bool
    normalize_const: str | None = None
    normalize_data: str | None = None
    sel_sims: list[int] | None = None
    trim_end: int = 0

    def __post_init__(self) -> None:
        self.num_sims = SIMULATION_COUNTS[self.dset_name]
        self.num_frames = FRAME_COUNTS[self.dset_name]
        self.samples_per_sim = self.num_frames - self.time_steps - self.trim_end

    def __len__(self) -> int:
        selected_count = len(self.sel_sims) if self.sel_sims is not None else self.num_sims
        return selected_count * self.samples_per_sim


def build_splits(dataset_name: str):
    original_dataset = ape_2d_xxl.PBDLDataset
    try:
        ape_2d_xxl.PBDLDataset = FakePBDLDataset
        return ape_2d_xxl.ape_2d_xxl_datasets(
            dataset_name=dataset_name,
            dataset_directory="/unused/official-data",
            unrolling_steps=1,
            test_unrolling_steps=29,
            normalize_data="mean-std",
            normalize_const="mean-std",
            dataset_profile="legacy_small",
        )
    finally:
        ape_2d_xxl.PBDLDataset = original_dataset


def test_all_official_splits() -> None:
    total_rollouts = 0
    for dataset_name in OFFICIAL_DATASETS:
        train, val, test = build_splits(dataset_name)
        train_sims, test_sims = ape_2d_xxl_simulation_split(
            dataset_name,
            "legacy_small",
        )
        test_dataset_name = (
            dataset_name + "_test"
            if dataset_name in SEPARATE_TEST_DATASETS
            else dataset_name
        )

        assert val.dataset is train.dataset
        assert test.dset_name == test_dataset_name
        assert test.sel_sims == test_sims
        assert test.samples_per_sim == 1

        if test_sims is None:
            expected_rollouts = SIMULATION_COUNTS[test_dataset_name]
        else:
            assert train.dataset.sel_sims == train_sims
            assert set(train.dataset.sel_sims).isdisjoint(test.sel_sims)
            expected_rollouts = len(test_sims)

        assert len(test) == expected_rollouts
        total_rollouts += expected_rollouts

    assert total_rollouts == 167


def main() -> None:
    test_all_official_splits()
    print("official test split smoke test passed")


if __name__ == "__main__":
    main()
