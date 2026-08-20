"""Plot one 7-column rollout comparison for each Full-256 PDE family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ROWS = (
    ("pde_ttt_s_ema", "PDE-TTT-S EMA"),
    ("published_pde_s", "Published PDE-S"),
    ("pde_ttt_s", "PDE-TTT-S"),
    ("pde_transformer_s", "PDE-Transformer-S"),
)

PDE_LABELS = {
    "diff": "Diffusion",
    "burgers": "Burgers",
    "kdv": "Korteweg-de Vries",
    "ks": "Kuramoto-Sivashinsky",
    "fisher": "Fisher-KPP",
    "gs_alpha": "Gray-Scott alpha",
    "gs_beta": "Gray-Scott beta",
    "gs_gamma": "Gray-Scott gamma",
    "gs_delta": "Gray-Scott delta",
    "gs_epsilon": "Gray-Scott epsilon",
    "gs_theta": "Gray-Scott theta",
    "gs_iota": "Gray-Scott iota",
    "gs_kappa": "Gray-Scott kappa",
    "sh": "Swift-Hohenberg",
    "decay_turb": "Decaying turbulence",
    "kolm_flow": "Kolmogorov flow",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _limits(arrays: list[np.ndarray]) -> tuple[float, float, str]:
    values = np.concatenate([array.reshape(-1) for array in arrays])
    low, high = np.nanpercentile(values, [1.0, 99.0])
    if low < 0 < high:
        limit = max(abs(float(low)), abs(float(high)), 1e-8)
        return -limit, limit, "RdBu_r"
    if high <= low:
        high = low + 1e-8
    return float(low), float(high), "viridis"


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for key, _ in MODEL_ROWS:
        manifest_path = input_dir / key / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"Missing model export: {manifest_path}")
        manifests[key] = json.loads(manifest_path.read_text(encoding="utf-8"))

    pdes = [record["pde"] for record in manifests[MODEL_ROWS[0][0]]["records"]]
    for key, _ in MODEL_ROWS[1:]:
        current = [record["pde"] for record in manifests[key]["records"]]
        if current != pdes:
            raise RuntimeError(f"PDE order differs for {key}: {current}")

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "figure.dpi": 160,
        }
    )
    index_rows = []
    for pde in pdes:
        exports = {key: _load(input_dir / key / f"{pde}.npz") for key, _ in MODEL_ROWS}
        reference = exports[MODEL_ROWS[0][0]]["reference"]
        steps = exports[MODEL_ROWS[0][0]]["steps"].astype(int)
        channel = int(exports[MODEL_ROWS[0][0]]["channel"])
        trajectory_index = int(exports[MODEL_ROWS[0][0]]["trajectory_index"])

        for key, _ in MODEL_ROWS[1:]:
            if not np.array_equal(exports[key]["steps"], steps):
                raise RuntimeError(f"Step mismatch for {pde}/{key}")
            if int(exports[key]["channel"]) != channel:
                raise RuntimeError(f"Channel mismatch for {pde}/{key}")
            if int(exports[key]["trajectory_index"]) != trajectory_index:
                raise RuntimeError(f"Trajectory mismatch for {pde}/{key}")
            if not np.allclose(exports[key]["reference"], reference, rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"Reference mismatch for {pde}/{key}")

        panel_arrays = [reference] + [exports[key]["prediction"] for key, _ in MODEL_ROWS]
        vmin, vmax, cmap = _limits(panel_arrays)
        fig, axes = plt.subplots(
            5,
            len(steps),
            figsize=(13.2, 8.1),
            constrained_layout=True,
            gridspec_kw={"wspace": 0.015, "hspace": 0.035},
        )
        row_names = ["Reference", *[label for _, label in MODEL_ROWS]]
        row_data = [reference, *[exports[key]["prediction"] for key, _ in MODEL_ROWS]]
        image = None
        for row, (row_name, data) in enumerate(zip(row_names, row_data)):
            for col, step in enumerate(steps):
                ax = axes[row, col]
                image = ax.imshow(data[col], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"Step {step}", pad=3)
                if col == 0:
                    ax.set_ylabel(row_name, rotation=0, ha="right", va="center", labelpad=8)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.4)
                    spine.set_color("#777777")

        assert image is not None
        fig.suptitle(PDE_LABELS.get(pde, pde), fontsize=12, fontweight="bold")
        colorbar = fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.022, pad=0.02)
        colorbar.ax.tick_params(labelsize=7)
        png_path = output_dir / f"{pde}.png"
        pdf_path = output_dir / f"{pde}.pdf"
        fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        index_rows.append(
            {
                "pde": pde,
                "trajectory_index": trajectory_index,
                "channel": channel,
                "steps": steps.tolist(),
                "vmin": vmin,
                "vmax": vmax,
                "colormap": cmap,
                "png": png_path.name,
                "pdf": pdf_path.name,
            }
        )
        print(f"wrote {png_path}")

    (output_dir / "index.json").write_text(
        json.dumps(index_rows, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
