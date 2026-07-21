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


SIMULATION_COUNTS = {
    "burgers": 60,
    "ks": 60,
    "ks_test": 5,
    "kolm_flow": 60,
    "kolm_flow_test": 5,
}

FRAME_COUNTS = {
    "burgers": 30,
    "ks": 30,
    "ks_test": 200,
    "kolm_flow": 30,
    "kolm_flow_test": 200,
}


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
        )
    finally:
        ape_2d_xxl.PBDLDataset = original_dataset


def test_joint_file_burgers_split() -> None:
    train, val, test = build_splits("burgers")

    assert train.dataset.sel_sims == list(range(0, 50))
    assert val.dataset is train.dataset
    assert test.dset_name == "burgers"
    assert test.sel_sims == list(range(50, 60))
    assert test.samples_per_sim == 1
    assert len(test) == 10
    assert set(train.dataset.sel_sims).isdisjoint(test.sel_sims)


def test_separate_ks_test_file() -> None:
    _, _, test = build_splits("ks")

    assert test.dset_name == "ks_test"
    assert test.sel_sims is None
    assert test.num_sims == 5
    assert test.samples_per_sim == 1
    assert len(test) == 5


def test_separate_kolmogorov_test_file() -> None:
    _, _, test = build_splits("kolm_flow")

    assert test.dset_name == "kolm_flow_test"
    assert test.sel_sims is None
    assert test.num_sims == 5
    assert test.samples_per_sim == 1
    assert len(test) == 5


def main() -> None:
    test_joint_file_burgers_split()
    test_separate_ks_test_file()
    test_separate_kolmogorov_test_file()
    print("official test split smoke test passed")


if __name__ == "__main__":
    main()
