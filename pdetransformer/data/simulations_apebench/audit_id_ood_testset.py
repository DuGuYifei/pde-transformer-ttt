"""Audit the deterministic APEBench ID/OOD test matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

try:
    from .generate_id_ood_testset import _dataset_completion_error
    from .id_ood_manifest import PDE_NAMES, simulation_entries
except ImportError:  # Allows direct execution without importing the model package.
    from generate_id_ood_testset import _dataset_completion_error
    from id_ood_manifest import PDE_NAMES, simulation_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid_total = 0
    invalid_total = 0
    missing_total = 0

    for pde in PDE_NAMES:
        path = args.output_dir / f"{pde}.hdf5"
        entries = simulation_entries(pde)
        valid = []
        invalid = []
        missing = []

        if path.exists():
            with h5py.File(path, "r") as handle:
                group = handle.get("sims")
                for entry in entries:
                    sim_id = int(entry["sim_id"])
                    name = f"sim{sim_id}"
                    if group is None or name not in group:
                        missing.append(sim_id)
                        continue
                    error = _dataset_completion_error(group[name], entry)
                    if error is None:
                        valid.append(sim_id)
                    else:
                        invalid.append((sim_id, error))
        else:
            missing = [int(entry["sim_id"]) for entry in entries]

        valid_total += len(valid)
        invalid_total += len(invalid)
        missing_total += len(missing)
        print(
            f"PDE_AUDIT pde={pde} valid={valid} missing={missing} invalid={invalid}",
            flush=True,
        )

    print(
        f"AUDIT_SUMMARY valid={valid_total} missing={missing_total} "
        f"invalid={invalid_total} expected=153",
        flush=True,
    )
    if invalid_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
