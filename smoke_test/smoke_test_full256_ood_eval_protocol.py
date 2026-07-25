from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = REPO_ROOT / "pretrained_eval" / "test_pretrained_mc_server.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("full256_ood_eval", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator from {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_manifest(evaluator) -> dict:
    seeds = [101, 102, 103]
    entries = []
    for condition_idx, condition in enumerate(evaluator.ID_OOD_CONDITIONS):
        for seed_idx, seed in enumerate(seeds):
            entries.append(
                {
                    "sim_id": condition_idx * 3 + seed_idx,
                    "condition": condition,
                    "seed": seed,
                    "parameter_overrides": {},
                    "numerical_overrides": {},
                }
            )
    return {
        "solver_resolution": 2048,
        "stored_resolution": 256,
        "evaluation_resolution": 128,
        "time_steps": 30,
        "rollout_transitions": 29,
        "conditions": list(evaluator.ID_OOD_CONDITIONS),
        "seeds": seeds,
        "pdes": {"diff": entries},
    }


def main() -> None:
    evaluator = load_evaluator()

    evaluator.validate_profile_test_split(
        "diff",
        "full_paper",
        {
            "source_dataset_name": "diff",
            "source_file_name": "diff.hdf5",
            "source_num_simulations": 600,
            "selected_sim_ids": list(range(500, 600)),
            "samples_per_simulation": 1,
        },
    )
    evaluator.validate_profile_test_split(
        "ks",
        "full_paper",
        {
            "source_dataset_name": "ks_test",
            "source_file_name": "ks_test.hdf5",
            "source_num_simulations": 50,
            "selected_sim_ids": list(range(50)),
            "samples_per_simulation": 1,
        },
    )

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "diff.hdf5").touch()
        (data_dir / "manifest.json").write_text(
            json.dumps(build_manifest(evaluator)),
            encoding="utf-8",
        )
        manifest = evaluator.load_id_ood_manifest(
            data_dir,
            ["diff"],
            sample_size=256,
            downsample_factor=1,
        )
        split_info = {
            "source_dataset_name": "diff",
            "source_file_name": "diff.hdf5",
            "source_num_frames": 30,
            "selected_sim_ids": list(range(9)),
            "samples_per_simulation": 1,
        }
        evaluator.validate_id_ood_test_split("diff", split_info, manifest)

    print("Full-256 and ID/OOD evaluator protocol smoke test passed.")


if __name__ == "__main__":
    main()
