from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = REPO_ROOT / "pretrained_eval" / "test_pretrained_mc_server.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("official_eval", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator from {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_splits_pass(evaluator) -> None:
    for pde, expectation in evaluator.STRICT_OFFICIAL_TEST_EXPECTATIONS.items():
        evaluator.validate_strict_official_test_split(
            pde,
            {
                "source_dataset_name": expectation["source_dataset_name"],
                "selected_sim_ids": expectation["selected_sim_ids"],
                "samples_per_simulation": 1,
            },
        )


def test_all_burgers_simulations_fail(evaluator) -> None:
    try:
        evaluator.validate_strict_official_test_split(
            "burgers",
            {
                "source_dataset_name": "burgers",
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
