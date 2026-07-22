"""CPU-only structural checks for the official-resolution ID/OOD generator."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SIMULATION_DIR = ROOT / "pdetransformer" / "data" / "simulations_apebench"
sys.path.insert(0, str(SIMULATION_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SIMULATION_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


manifest_module = _load("id_ood_manifest", "id_ood_manifest.py")
generator_module = _load("generate_id_ood_testset", "generate_id_ood_testset.py")


def main() -> None:
    manifest = manifest_module.build_manifest()
    assert manifest["solver_resolution"] == 2048
    assert manifest["stored_resolution"] == 256
    assert manifest["evaluation_resolution"] == 128
    assert len(manifest["pdes"]) == 17
    assert sum(len(entries) for entries in manifest["pdes"].values()) == 153

    for pde, entries in manifest["pdes"].items():
        assert [entry["sim_id"] for entry in entries] == list(range(9)), pde
        for condition in manifest_module.CONDITIONS:
            selected = [entry for entry in entries if entry["condition"] == condition]
            assert [entry["seed"] for entry in selected] == list(manifest_module.SEEDS), pde
        assert min(entry["seed"] for entry in entries) >= 2**30

    for pde, conditions in manifest_module.PARAMETER_MATRIX.items():
        assert set(conditions) == set(manifest_module.CONDITIONS), pde
        assert conditions["id"] != conditions["ood_low"], pde
        assert conditions["id"] != conditions["ood_high"], pde

    coordinates = np.arange(2048, dtype=np.float32)
    frame = coordinates[:, None] + coordinates[None, :]
    pooled = generator_module._downsample_2048_to_256(frame)
    assert pooled.shape == (256, 256)
    np.testing.assert_allclose(pooled[0, 0], 7.0)
    np.testing.assert_allclose(pooled[-1, -1], 4087.0)

    setup_source = (SIMULATION_DIR / "simulation_setups_2d.py").read_text(encoding="utf-8")
    setup_tree = ast.parse(setup_source)
    setup_function = next(
        node for node in setup_tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_setup_2d"
    )
    argument_names = [argument.arg for argument in setup_function.args.args]
    assert argument_names[-2:] == ["seed_override", "parameter_overrides"]
    assert '"Resolution": 2048' in setup_source

    print("ID/OOD generator smoke test passed: 17 PDEs, 153 trajectories, 2048 -> 256.")


if __name__ == "__main__":
    main()
