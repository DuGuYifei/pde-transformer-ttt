"""Generate thesis result figures from retained machine-readable evidence."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
HISTORY = HERE.parents[2] / "train-history"
FULL = HISTORY / "global-linear-full256-evaluation"
RAW42 = HISTORY / "global-linear-s-rawbest-3seed-full256-evaluation" / "seed42"
SEEDS = HISTORY / "global-linear-s-ema-3seed-full256-evaluation" / "summary_3seed"
MATCHED_METRICS = FULL / "p_vs_global_linear_training_metrics.csv"
RAW42_TRAINING_METRICS = HERE / "data" / "pde_ttt_s_seed42_ema_enabled_training_metrics.csv"
FULL_ALL29_STATS = HERE / "data" / "full256_all29_family_statistics.csv"

BLUE = "#0065BD"
ORANGE = "#E37222"
GREEN = "#A2AD00"
DARK_BLUE = "#003359"
GRAY = "#666666"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 240,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(HERE / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture_screen() -> None:
    rows = [
        ("PDE-Transformer-S (33.19M)", 338.3, 1.0208, 33.19, "Baseline"),
        ("Window TTT-Linear (33.31M)", 313.2, 1.0301, 33.31, "Window TTT"),
        ("Window TTT-MLP (34.55M)", 406.3, 0.9860, 34.55, "Window TTT"),
        ("Window Sequential TTT-Linear (33.31M)", 370.9, 0.9651, 33.31, "Window TTT"),
        ("Window Sequential TTT-MLP (34.55M)", 574.1, 0.9980, 34.55, "Window TTT"),
        ("Window ViT$^3$ (37.23M)", 429.0, 0.9858, 37.23, "Window TTT"),
        ("Window Attention--TTT Hybrid (43.12M)", 967.6, 1.0623, 43.12, "Hybrid"),
        ("Full-map ViT$^3$ (36.13M)", 460.4, 0.9635, 36.13, "Nonlinear full map"),
        ("PDE-TTT-S (33.36M)", 329.8, 0.9443, 33.36, "Linear full map"),
    ]
    colors = {
        "Baseline": GRAY,
        "Window TTT": ORANGE,
        "Hybrid": GREEN,
        "Nonlinear full map": DARK_BLUE,
        "Linear full map": BLUE,
    }
    offsets = {
        "PDE-Transformer-S (33.19M)": (7, -12),
        "Window TTT-Linear (33.31M)": (7, 12),
        "Window TTT-MLP (34.55M)": (-8, 13),
        "Window Sequential TTT-Linear (33.31M)": (7, -14),
        "Window Sequential TTT-MLP (34.55M)": (7, 12),
        "Window ViT$^3$ (37.23M)": (6, -9),
        "Window Attention--TTT Hybrid (43.12M)": (-150, 7),
        "Full-map ViT$^3$ (36.13M)": (5, 7),
        "PDE-TTT-S (33.36M)": (5, 7),
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    for label, time_s, error, params, family in rows:
        size = 22 + (params - 30) * 12
        ax.scatter(
            time_s,
            error,
            s=size,
            marker="o",
            facecolor=colors[family],
            edgecolor=colors[family],
            linewidth=1.5,
            zorder=3,
        )
        dx, dy = offsets[label]
        right_aligned = label in {
            "Window TTT-MLP (34.55M)",
        }
        ax.annotate(
            label,
            (time_s, error),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            ha="right" if right_aligned else "left",
            va="center",
        )
    ax.set_xscale("log")
    ax.set_xlabel("End-to-end 29-step evaluation time (s, log scale)")
    ax.set_ylabel("Step-20 development nRMSE")
    ax.grid(True, which="both", color="#dddddd", linewidth=0.6, alpha=0.8)
    family_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=c, markerfacecolor=c, label=k)
        for k, c in colors.items()
    ]
    ax.set_ylim(0.925, 1.075)
    ax.legend(handles=family_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.text(0.01, 0.96, "Exploratory screen; mean over 17 PDE families", transform=ax.transAxes, fontsize=7, color=GRAY)
    save(fig, "architecture_screen_tradeoff.png")


def matched_training_curves() -> None:
    baseline = pd.read_csv(MATCHED_METRICS)
    baseline = baseline.loc[baseline["model"] == "P (Plain Attention)"].copy()
    # The retained baseline summary uses completed-epoch numbering (1--100).
    # Convert it to the zero-based checkpoint epoch index used by Lightning.
    baseline["epoch"] = baseline["epoch"] - 1
    baseline["label"] = "PDE-Transformer"

    raw_metrics = pd.read_csv(RAW42_TRAINING_METRICS)
    raw_train = raw_metrics.loc[raw_metrics["loss_epoch"].notna(), ["epoch", "loss_epoch"]]
    raw_val = raw_metrics.loc[
        raw_metrics["val/raw_loss_epoch"].notna(), ["epoch", "val/raw_loss_epoch"]
    ]
    pde_ttt = raw_train.merge(raw_val, on="epoch", validate="one_to_one").rename(
        columns={"loss_epoch": "train_loss", "val/raw_loss_epoch": "val_loss"}
    )
    pde_ttt["label"] = "PDE-TTT"

    df = pd.concat(
        [
            baseline[["epoch", "train_loss", "val_loss", "label"]],
            pde_ttt[["epoch", "train_loss", "val_loss", "label"]],
        ],
        ignore_index=True,
    )
    colors = {
        "PDE-Transformer": ORANGE,
        "PDE-TTT": BLUE,
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.45), sharex=True)
    for label in ("PDE-Transformer", "PDE-TTT"):
        rows = df[df["label"] == label].sort_values("epoch")
        if rows.empty:
            raise ValueError(f"Missing matched-run metrics for {label}")
        color = colors[label]
        axes[0].plot(rows["epoch"], rows["train_loss"], color=color, linewidth=1.7, label=label)
        axes[1].plot(rows["epoch"], rows["val_loss"], color=color, linewidth=1.7, label=label)

        selected = rows.loc[rows["val_loss"].idxmin()]
        selected_epoch = float(selected["epoch"])
        for axis in axes:
            axis.axvline(selected_epoch, color=color, linestyle=":", linewidth=1.1, alpha=0.9)
        axes[1].scatter(
            [selected_epoch],
            [float(selected["val_loss"])],
            s=24,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=4,
        )

    for axis, title in zip(axes, ("Training loss", "Validation loss")):
        axis.set_title(title)
        axis.set_xlabel("Checkpoint epoch index")
        axis.set_ylabel("MSE loss")
        axis.set_yscale("log")
        axis.set_xlim(0, 99)
        axis.grid(True, which="both", color="#dddddd", linewidth=0.6, alpha=0.8)
        axis.legend(frameon=False, loc="upper right")

    fig.suptitle("PDE-S Full-256: PDE-Transformer versus PDE-TTT", y=1.02)
    fig.tight_layout()
    save(fig, "matched_pde_transformer_vs_pde_ttt_training_curves.png")


def aggregate_rollout() -> None:
    steps = np.arange(1, 30)
    stats = pd.read_csv(FULL_ALL29_STATS)

    def baseline_statistics() -> tuple[np.ndarray, np.ndarray]:
        rows = stats.loc[stats["model"] == "PDE-Transformer-S"].sort_values("step")
        if rows["step"].tolist() != steps.tolist():
            raise ValueError("Expected rollout steps 1--29 for PDE-Transformer-S")
        return rows["mean_nRMSE"].to_numpy(), rows["sample_sd_across_pde_families"].to_numpy()

    def pde_ttt_statistics() -> tuple[np.ndarray, np.ndarray]:
        rows = pd.read_csv(RAW42 / "full_test" / "results_cache_off.csv")
        values = rows[[f"nRMSE_{step}" for step in steps]]
        return values.mean(axis=0).to_numpy(), values.std(axis=0, ddof=1).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    p, p_sd = baseline_statistics()
    g, g_sd = pde_ttt_statistics()
    axes[0].plot(steps, p, color=ORANGE, linewidth=1.8, label="PDE-Transformer-S")
    axes[0].plot(steps, g, color=BLUE, linewidth=1.8, label="PDE-TTT-S")
    axes[0].scatter(steps, p, color=ORANGE, s=8, zorder=3)
    axes[0].scatter(steps, g, color=BLUE, s=8, zorder=3)
    axes[0].fill_between(steps, np.maximum(0, p - p_sd), p + p_sd, color=ORANGE, alpha=0.16, linewidth=0)
    axes[0].fill_between(steps, np.maximum(0, g - g_sd), g + g_sd, color=BLUE, alpha=0.16, linewidth=0)
    axes[0].set_title("Mean over 16 PDE families")
    axes[0].set_xlabel("Rollout step")
    axes[0].set_ylabel("nRMSE")
    axes[0].set_xticks([1, 5, 10, 15, 20, 25, 29])
    axes[0].set_xlim(1, 29)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, color="#dddddd", linewidth=0.6)
    reduction = 100 * (p - g) / p
    axes[1].plot(steps, reduction, color=BLUE, linewidth=1.8, marker="o", markersize=2.8)
    axes[1].fill_between(steps, 0, reduction, color=BLUE, alpha=0.12)
    axes[1].set_title("PDE-TTT-S error reduction")
    axes[1].set_xlabel("Rollout step")
    axes[1].set_ylabel("Relative reduction (%)")
    axes[1].set_xticks([1, 5, 10, 15, 20, 25, 29])
    axes[1].set_xlim(1, 29)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].grid(True, axis="y", color="#dddddd", linewidth=0.6)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "full256_reported_horizon_rollout.png")


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


def per_pde_horizon_heatmap() -> None:
    steps = [1, 10, 20, 29]
    p = pd.read_csv(FULL / "plain_attention" / "full_test" / "results_cache_off.csv").set_index("dataset")
    g = pd.read_csv(RAW42 / "full_test" / "results_cache_off.csv").set_index("dataset")
    order = [name for name in g.index if name in PDE_LABELS and name != "hyp"]
    values = 100 * np.array([[(g.loc[name, f"nRMSE_{s}"] - p.loc[name, f"nRMSE_{s}"]) / p.loc[name, f"nRMSE_{s}"] for s in steps] for name in order])
    limit = max(abs(np.nanpercentile(values, 2)), abs(np.nanpercentile(values, 98)))
    fig, ax = plt.subplots(figsize=(6.2, 6.1))
    im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(steps)), [f"@{s}" for s in steps])
    ax.set_yticks(range(len(order)), [PDE_LABELS.get(x, x) for x in order], fontsize=7.5)
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("PDE family")
    for row in range(len(order)):
        for col in range(len(steps)):
            rounded = int(np.rint(values[row, col]))
            label = "0" if rounded == 0 else f"{rounded:+d}"
            ax.text(col, row, label, ha="center", va="center", fontsize=6.5, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Relative nRMSE change (%)\nPDE-TTT vs. PDE-Transformer")
    fig.tight_layout()
    save(fig, "per_pde_reported_horizon_heatmap.png")


def seed_uncertainty() -> None:
    df = pd.read_csv(SEEDS / "per_pde_per_model_seed.csv")
    # Hyper-diffusion was not part of Full-256 training. Keep the figure on
    # the same 16-family basis as the corresponding summary table.
    df = df[df["pde"] != "hyp"].copy()
    steps = np.arange(1, 30)
    panels = [
        ("full_test", "all", "Strict Full-256"),
        ("id_ood", "id", "Generated ID"),
        ("id_ood", "ood_low", "Generated OOD-low"),
        ("id_ood", "ood_high", "Generated OOD-high"),
    ]
    seed_colors = {42: BLUE, 43: ORANGE, 44: GREEN}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex=True, sharey=True)
    for ax, (split, condition, title) in zip(axes.flat, panels):
        rows = df[(df["split"] == split) & (df["condition"] == condition)]
        curves = []
        for model_seed in (42, 43, 44):
            seed_rows = rows[rows["model_seed"] == model_seed]
            curve = seed_rows[
                [f"nRMSE_{step}" for step in steps]
            ].mean(axis=0).to_numpy()
            curves.append(curve)
            ax.plot(
                steps,
                curve,
                color=seed_colors[model_seed],
                alpha=0.95,
                linewidth=1.3,
                label=f"Seed {model_seed}",
            )
        curves = np.stack(curves)
        mean = curves.mean(axis=0)
        sample_sd = curves.std(axis=0, ddof=1)
        ax.fill_between(
            steps,
            mean - sample_sd,
            mean + sample_sd,
            color=BLUE,
            alpha=0.15,
            linewidth=0,
            label=r"Mean $\pm$ sample SD",
        )
        ax.plot(steps, mean, color="#222222", linewidth=2.0, label="Three-seed mean")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 29])
        ax.grid(True, color="#dddddd", linewidth=0.6)
    for ax in axes[-1, :]:
        ax.set_xlabel("Rollout step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean nRMSE over 16 PDE families")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [0, 1, 2, 4, 3]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "three_seed_rollout_uncertainty.png")


def parameter_shift_matrix() -> None:
    steps = [1, 10, 20, 29]
    conditions = ["id", "ood_low", "ood_high"]
    p = pd.read_csv(FULL / "plain_attention" / "id_ood" / "results_conditions_cache_off.csv")
    g = pd.read_csv(RAW42 / "id_ood" / "results_conditions_cache_off.csv")
    order = [name for name in dict.fromkeys(g["dataset"].tolist()) if name in PDE_LABELS]
    matrix = []
    for name in order:
        row = []
        for condition in conditions:
            p_row = p[(p.dataset == name) & (p.condition == condition)].iloc[0]
            g_row = g[(g.dataset == name) & (g.condition == condition)].iloc[0]
            row.extend([g_row[f"nRMSE_{s}"] - p_row[f"nRMSE_{s}"] for s in steps])
        matrix.append(row)
    matrix = np.asarray(matrix)
    limit = max(abs(np.nanpercentile(matrix, 2)), abs(np.nanpercentile(matrix, 98)))
    fig, ax = plt.subplots(figsize=(8.7, 6.0))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    labels = [f"{c.replace('_', '-')} @{s}" for c in conditions for s in steps]
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(order)), [PDE_LABELS.get(x, x) for x in order], fontsize=7.5)
    for boundary in [3.5, 7.5]:
        ax.axvline(boundary, color="black", linewidth=1)
    if "hyp" in order:
        row = order.index("hyp")
        ax.add_patch(Rectangle((-0.5, row - 0.5), len(labels), 1, fill=False, edgecolor="black", linewidth=1.5))
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("nRMSE difference\nPDE-TTT minus PDE-Transformer")
    ax.set_xlabel("Condition and rollout horizon")
    ax.set_ylabel("PDE family")
    fig.tight_layout()
    save(fig, "parameter_shift_response_matrix.png")


if __name__ == "__main__":
    architecture_screen()
    matched_training_curves()
    aggregate_rollout()
    per_pde_horizon_heatmap()
    seed_uncertainty()
    parameter_shift_matrix()
