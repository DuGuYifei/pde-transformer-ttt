from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUNS = {
    "P (Plain Attention)": ROOT / "plain_attention" / "training.log",
    "G-L (Global Linear TTT)": ROOT / "training.log",
}

TRAIN_RE = re.compile(
    r"\[train\] epoch=(?P<epoch>\d+)/100 .*?"
    r"elapsed=(?P<elapsed>[0-9.]+)min loss=(?P<loss>[0-9.eE+-]+)"
)
VAL_RE = re.compile(
    r"\[val\] epoch=(?P<epoch>\d+) loss=(?P<loss>[0-9.eE+-]+)"
)


def parse_log(path: Path) -> list[dict[str, float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    train = {
        int(match.group("epoch")): (
            float(match.group("loss")),
            float(match.group("elapsed")),
        )
        for match in TRAIN_RE.finditer(text)
    }
    validation = {
        int(match.group("epoch")): float(match.group("loss"))
        for match in VAL_RE.finditer(text)
        if int(match.group("epoch")) <= 100
    }
    epochs = sorted(set(train) & set(validation))
    if epochs != list(range(1, 101)):
        raise RuntimeError(f"{path}: expected epochs 1..100, got {epochs}")
    return [
        {
            "epoch": epoch,
            "train_loss": train[epoch][0],
            "val_loss": validation[epoch],
            "elapsed_minutes": train[epoch][1],
        }
        for epoch in epochs
    ]


def main() -> None:
    parsed = {name: parse_log(path) for name, path in RUNS.items()}

    csv_path = ROOT / "p_vs_global_linear_training_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "epoch",
                "train_loss",
                "val_loss",
                "elapsed_minutes",
            ],
        )
        writer.writeheader()
        for name, rows in parsed.items():
            for row in rows:
                writer.writerow({"model": name, **row})

    colors = {
        "P (Plain Attention)": "#C64B40",
        "G-L (Global Linear TTT)": "#267A68",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for name, rows in parsed.items():
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(
            epochs,
            [row["train_loss"] for row in rows],
            color=colors[name],
            linewidth=2,
            label=name,
        )
        axes[1].plot(
            epochs,
            [row["val_loss"] for row in rows],
            color=colors[name],
            linewidth=2,
            label=name,
        )

    for axis, title, ylabel in (
        (axes[0], "Training loss", "MSE loss"),
        (axes[1], "Validation loss", "MSE loss"),
    ):
        axis.set_title(title)
        axis.set_xlabel("Completed epoch")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)

    fig.suptitle("PDE-S Full-256: Plain Attention vs Global Linear TTT")
    fig.savefig(ROOT / "p_vs_global_linear_training_curves.png", dpi=180)
    plt.close(fig)

    for name, rows in parsed.items():
        average = sum(row["elapsed_minutes"] for row in rows) / len(rows)
        total = sum(row["elapsed_minutes"] for row in rows)
        best = min(rows, key=lambda row: row["val_loss"])
        print(
            f"{name}: avg={average:.3f} min/epoch, total={total:.1f} min, "
            f"best logged val={best['val_loss']:.8g} at completed epoch "
            f"{int(best['epoch'])}"
        )


if __name__ == "__main__":
    main()
