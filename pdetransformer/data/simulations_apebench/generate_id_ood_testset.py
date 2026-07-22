"""Generate the deterministic APEBench ID/OOD test matrix at 2048 -> 256."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from .id_ood_manifest import PDE_NAMES, build_manifest, simulation_entries
except ImportError:  # Allows direct execution without importing the full pdetransformer package.
    from id_ood_manifest import PDE_NAMES, build_manifest, simulation_entries


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _downsample_2048_to_256(frame):
    import numpy as np

    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape[-2:] != (2048, 2048):
        raise ValueError(f"Expected a 2048x2048 frame, received {frame.shape}")
    leading = frame.shape[:-2]
    pooled = frame.reshape(*leading, 256, 8, 256, 8).mean(axis=(-3, -1), dtype=np.float32)
    if pooled.shape[-2:] != (256, 256):
        raise AssertionError(f"Unexpected pooled shape {pooled.shape}")
    return pooled


def _initialize_hdf5(path: Path, fixed: dict[str, Any]) -> None:
    import h5py

    if path.exists():
        return
    with h5py.File(path, "w") as handle:
        group = handle.create_group("sims", track_order=True)
        for key, value in fixed.items():
            group.attrs[key] = value
        group.attrs["Stored Resolution"] = 256
        group.attrs["Generation Protocol"] = "solve_2048_then_average_pool_8x8"


def _completed_simulations(path: Path) -> set[int]:
    import h5py

    if not path.exists():
        return set()
    with h5py.File(path, "r") as handle:
        return {
            int(name.removeprefix("sim"))
            for name in handle["sims"].keys()
            if name.startswith("sim") and name.removeprefix("sim").isdigit()
        }


def _write_simulation(path: Path, sim_id: int, data, fixed, varied, entry) -> None:
    import h5py

    with h5py.File(path, "a") as handle:
        group = handle["sims"]
        name = f"sim{sim_id}"
        if name in group:
            return
        dataset = group.create_dataset(name, data=data)
        for key in fixed["Constants"]:
            dataset.attrs[key] = varied[key]
        dataset.attrs["Seed"] = int(entry["seed"])
        dataset.attrs["Condition"] = entry["condition"]
        dataset.attrs["Solver Resolution"] = 2048
        dataset.attrs["Stored Resolution"] = 256


def generate_pde(pde: str, output_dir: Path) -> None:
    import exponax as ex
    import jax
    import jax.numpy as jnp
    import numpy as np

    try:
        from .simulation_setups_2d import get_setup_2d
    except ImportError:
        from simulation_setups_2d import get_setup_2d

    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_dir / f"{pde}.hdf5"
    metadata_path = output_dir / f"{pde}.json"
    metadata = {"protocol": build_manifest(), "simulations": {}}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    complete = _completed_simulations(hdf5_path)
    for entry in simulation_entries(pde):
        sim_id = int(entry["sim_id"])
        if sim_id in complete:
            print(f"SIM_SKIP pde={pde} sim={sim_id}", flush=True)
            continue

        started = time.perf_counter()
        fixed, varied, stepper, state = get_setup_2d(
            pde,
            False,
            sim_id,
            seed_override=entry["seed"],
            parameter_overrides=entry["parameter_overrides"],
        )
        if int(fixed["Resolution"]) != 2048:
            raise AssertionError(f"{pde} unexpectedly uses resolution {fixed['Resolution']}")
        for key, expected in entry["parameter_overrides"].items():
            if key not in varied:
                raise KeyError(f"{pde} did not apply parameter override {key!r}")
            if not np.allclose(varied[key], expected, rtol=0.0, atol=1.0e-12):
                raise AssertionError(
                    f"{pde} override {key!r} expected {expected}, received {varied[key]}"
                )
        _initialize_hdf5(hdf5_path, fixed)

        repeated_stepper = ex.RepeatedStepper(stepper, int(fixed["Sub Steps"]))
        warmup_steps = int(fixed.get("Warmup Steps", 0))
        total_steps = int(fixed["Time Steps"]) + warmup_steps
        frames = []
        for step in range(total_steps):
            state = repeated_stepper(state)
            if bool(jnp.isnan(state).any()):
                raise FloatingPointError(f"NaN in {pde} sim{sim_id} at integration step {step}")
            if step >= warmup_steps:
                frames.append(_downsample_2048_to_256(state))

        data = np.stack(frames, axis=0)
        if data.shape[0] != 30:
            raise AssertionError(f"{pde} sim{sim_id} produced {data.shape[0]} saved frames")
        _write_simulation(hdf5_path, sim_id, data, fixed, varied, entry)

        elapsed = time.perf_counter() - started
        metadata["simulations"][f"sim_{sim_id:04d}"] = {
            **entry,
            "actual_parameters": _jsonable(varied),
            "shape": list(data.shape),
            "elapsed_seconds": elapsed,
        }
        _write_json_atomic(metadata_path, metadata)
        print(
            f"SIM_DONE pde={pde} sim={sim_id} condition={entry['condition']} "
            f"seed={entry['seed']} seconds={elapsed:.3f}",
            flush=True,
        )
        del frames, data, state, repeated_stepper, stepper
        gc.collect()
        complete.add(sim_id)

    jax.clear_caches()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pdes", nargs="+", default=list(PDE_NAMES))
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--write-manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.pdes) - set(PDE_NAMES))
    if unknown:
        raise ValueError(f"Unknown PDEs: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.exists():
        _write_json_atomic(manifest_path, build_manifest())
    if args.write_manifest_only:
        print(manifest_path)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")
    for pde in args.pdes:
        generate_pde(pde, args.output_dir)


if __name__ == "__main__":
    main()
