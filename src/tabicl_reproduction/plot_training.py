"""Render recorded training and held-out validation loss; never invent results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (args.run / "history.jsonl").read_text().splitlines()
    ]
    train = [row for row in rows if row["kind"] == "train"]
    val = [row for row in rows if row["kind"] == "validation"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    axes[0].plot(
        [r["step"] for r in train],
        [r["loss"] for r in train],
        alpha=0.45,
        label="Training tasks",
    )
    axes[0].plot(
        [r["step"] for r in val],
        [r["log_loss_known"] for r in val],
        "o-",
        label="Fixed validation tasks",
    )
    axes[0].set(
        xlabel="Optimizer step",
        ylabel="Cross-entropy / log loss",
        title="Learning curves (known labels)",
    )
    axes[0].legend()
    axes[1].plot([r["step"] for r in val], [r["accuracy"] for r in val], "o-")
    axes[1].set(
        xlabel="Optimizer step",
        ylabel="Mean table accuracy",
        ylim=(0, 1),
        title="Independent synthetic validation",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(args.run / "learning_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
