from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = REPO_ROOT / "pretrained_eval" / "test_pretrained_mc_server.py"
SOURCE_SIMULATION_COUNTS = {
    "gs_alpha_test": 3,
    "gs_beta_test": 3,
    "gs_gamma_test": 3,
    "gs_epsilon_test": 3,
    "ks_test": 5,
    "decay_turb_test": 5,
    "kolm_flow_test": 5,
    "gs_delta": 100,
    "gs_theta": 100,
    "gs_iota": 100,
    "gs_kappa": 100,
}


def load_evaluator():
    spec = importlib.util.spec_from_file_location("official_eval", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator from {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_splits_pass(evaluator) -> None:
    total_rollouts = 0
    for pde in evaluator.DATASET_NAMES:
        split_spec = evaluator.ape_2d_xxl_split_spec(pde)
        source_name = split_spec["test_dataset_name"]
        source_num_simulations = SOURCE_SIMULATION_COUNTS.get(source_name, 60)
        selected_sim_ids = split_spec["test_sims"]
        if selected_sim_ids is None:
            selected_sim_ids = list(range(source_num_simulations))
        evaluator.validate_strict_official_test_split(
            pde,
            {
                "source_dataset_name": source_name,
                "source_file_name": source_name + ".hdf5",
                "source_num_simulations": source_num_simulations,
                "source_num_frames": evaluator.STRICT_OFFICIAL_SOURCE_FRAMES[pde],
                "selected_sim_ids": selected_sim_ids,
                "samples_per_simulation": 1,
            },
        )
        total_rollouts += len(selected_sim_ids)

    assert set(evaluator.DATASET_NAMES) == evaluator.STRICT_OFFICIAL_TEST_DATASETS
    assert total_rollouts == 167


def test_all_burgers_simulations_fail(evaluator) -> None:
    try:
        evaluator.validate_strict_official_test_split(
            "burgers",
            {
                "source_dataset_name": "burgers",
                "source_file_name": "burgers.hdf5",
                "source_num_simulations": 60,
                "source_num_frames": 30,
                "selected_sim_ids": list(range(60)),
                "samples_per_simulation": 1,
            },
        )
    except RuntimeError:
        return
    raise AssertionError("Strict validation accepted all 60 Burgers simulations")


def main() -> None:
    evaluator = load_evaluator()
    test_expected_splits_pass(evaluator)
    test_all_burgers_simulations_fail(evaluator)
    print("official evaluator protocol smoke test passed")


if __name__ == "__main__":
    main()
