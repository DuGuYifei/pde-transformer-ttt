"""Plot raw three-seed PDE-TTT-S results against the matched raw baseline."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
HISTORY = HERE.parents[1]
RAW_SEEDS = HERE / "per_pde_per_model_seed.csv"
PLAIN = (
    HISTORY
    / "global-linear-full256-evaluation"
    / "plain_attention"
    / "id_ood"
    / "results_conditions_cache_off.csv"
)

STEPS = [1, 10, 20, 29]
CONDITIONS = ["id", "ood_low", "ood_high"]
PDE_LABELS = {
    "diff": "Diffusion",
    "hyp": "Hyper-diffusion",
    "burgers": "Burgers",
    "kdv": "Korteweg--de Vries",
    "ks": "Kuramoto--Sivashinsky",
    "fisher": "Fisher--KPP",
    "gs_alpha": "Gray--Scott alpha",
    "gs_beta": "Gray--Scott beta",
    "gs_gamma": "Gray--Scott gamma",
    "gs_delta": "Gray--Scott delta",
    "gs_epsilon": "Gray--Scott epsilon",
    "gs_theta": "Gray--Scott theta",
    "gs_iota": "Gray--Scott iota",
    "gs_kappa": "Gray--Scott kappa",
    "sh": "Swift--Hohenberg",
    "decay_turb": "Decaying turbulence",
    "kolm_flow": "Kolmogorov flow",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 240,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def build_matrices() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_csv(RAW_SEEDS)
    raw = raw[(raw["split"] == "id_ood") & raw["pde"].isin(PDE_LABELS)]
    plain = pd.read_csv(PLAIN)
    order = [name for name in dict.fromkeys(plain["dataset"]) if name in PDE_LABELS]

    mean_difference = []
    seed_std = []
    winning_seed_count = []
    records = []

    for pde in order:
        mean_row = []
        std_row = []
        wins_row = []
        for condition in CONDITIONS:
            baseline_row = plain[
                (plain["dataset"] == pde) & (plain["condition"] == condition)
            ].iloc[0]
            seed_rows = raw[(raw["pde"] == pde) & (raw["condition"] == condition)]
            if len(seed_rows) != 3:
                raise ValueError(f"Expected three raw model seeds for {pde}/{condition}")
            for step in STEPS:
                baseline = float(baseline_row[f"nRMSE_{step}"])
                values = seed_rows[f"nRMSE_{step}"].to_numpy(dtype=float)
                difference = values - baseline
                mean_row.append(float(difference.mean()))
                std_row.append(float(difference.std(ddof=1)))
                wins_row.append(int(np.sum(difference < 0)))
                records.append(
                    {
                        "pde": pde,
                        "condition": condition,
                        "step": step,
                        "pde_transformer_raw": baseline,
                        "pde_ttt_raw_3seed_mean": float(values.mean()),
                        "pde_ttt_raw_3seed_std": float(values.std(ddof=1)),
                        "mean_difference": float(difference.mean()),
                        "winning_seed_count": int(np.sum(difference < 0)),
                    }
                )
        mean_difference.append(mean_row)
        seed_std.append(std_row)
        winning_seed_count.append(wins_row)

    pd.DataFrame(records).to_csv(HERE / "raw_3seed_parameter_shift_values.csv", index=False)
    return (
        order,
        np.asarray(mean_difference),
        np.asarray(seed_std),
        np.asarray(winning_seed_count),
    )


def decorate_axis(ax: plt.Axes, order: list[str]) -> None:
    labels = [f"{condition.replace('_', '-')} @{step}" for condition in CONDITIONS for step in STEPS]
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(order)), [PDE_LABELS[name] for name in order], fontsize=7.5)
    for boundary in [3.5, 7.5]:
        ax.axvline(boundary, color="black", linewidth=1)
    if "hyp" in order:
        row = order.index("hyp")
        ax.add_patch(
            Rectangle(
                (-0.5, row - 0.5),
                len(labels),
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
        )
    ax.set_xlabel("Condition and rollout horizon")


def plot_mean(order: list[str], mean_difference: np.ndarray) -> None:
    limit = max(
        abs(np.nanpercentile(mean_difference, 2)),
        abs(np.nanpercentile(mean_difference, 98)),
    )
    fig, ax = plt.subplots(figsize=(8.7, 6.0))
    im = ax.imshow(mean_difference, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    decorate_axis(ax, order)
    ax.set_ylabel("PDE family")
    ax.set_title("Raw PDE-TTT-S three-seed mean minus raw PDE-Transformer-S")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean nRMSE difference\nPDE-TTT-S minus PDE-Transformer-S")
    fig.tight_layout()
    fig.savefig(HERE / "raw_3seed_parameter_shift_response_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic(
    order: list[str],
    mean_difference: np.ndarray,
    winning_seed_count: np.ndarray,
) -> None:
    limit = max(
        abs(np.nanpercentile(mean_difference, 2)),
        abs(np.nanpercentile(mean_difference, 98)),
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 11.0), sharex=True)
    mean_im = axes[0].imshow(
        mean_difference,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    decorate_axis(axes[0], order)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("PDE family")
    axes[0].set_title("(a) Mean raw nRMSE difference across model seeds 42, 43, and 44")
    cbar = fig.colorbar(mean_im, ax=axes[0], fraction=0.025, pad=0.02)
    cbar.set_label("PDE-TTT-S minus PDE-Transformer-S")

    wins_im = axes[1].imshow(
        winning_seed_count,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=3,
    )
    decorate_axis(axes[1], order)
    axes[1].set_ylabel("PDE family")
    axes[1].set_title("(b) Raw PDE-TTT-S model seeds with lower nRMSE than the baseline")
    for row in range(winning_seed_count.shape[0]):
        for col in range(winning_seed_count.shape[1]):
            count = int(winning_seed_count[row, col])
            axes[1].text(
                col,
                row,
                f"{count}/3",
                ha="center",
                va="center",
                fontsize=6.3,
                color="white" if count >= 2 else "black",
            )
    cbar = fig.colorbar(wins_im, ax=axes[1], fraction=0.025, pad=0.02, ticks=[0, 1, 2, 3])
    cbar.set_label("Winning model seeds")
    fig.tight_layout()
    fig.savefig(HERE / "raw_3seed_parameter_shift_mean_and_consistency.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pde_order, differences, standard_deviations, wins = build_matrices()
    plot_mean(pde_order, differences)
    plot_diagnostic(pde_order, differences, wins)
    print(f"Wrote figures and values to {HERE}")
