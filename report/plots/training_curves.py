"""Training loss and eval log loss over the steps of one run.

    uv run python report/plots/training_curves.py <run folder>

The run folder is the one holding metrics.jsonl (outputs/runs/<date>/<time>/).
Writes report/figures/training_curves.pdf.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from style import COLORS, TEXT_WIDTH_IN, save, use_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--smooth", type=int, default=20, help="rows to average")
    args = parser.parse_args()

    metrics = pd.read_json(args.run_dir / "metrics.jsonl", lines=True)
    train = metrics.dropna(subset=["train/loss"])
    evals = metrics.dropna(subset=["eval/log_loss"])

    use_style()
    # sharex: both panels use the same step range, so they line up.
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 1.9), sharex=True)

    # Raw values are noisy (one window of tables each); show them faintly
    # behind a moving average.
    left.plot(train["step"], train["train/loss"], color=COLORS[0], alpha=0.2)
    smooth = train["train/loss"].rolling(args.smooth, min_periods=1).mean()
    left.plot(train["step"], smooth, color=COLORS[0], label="moving average")
    left.set(xlabel="Optimizer step", ylabel="Training loss (prior)")
    left.legend()

    right.plot(evals["step"], evals["eval/log_loss"], color=COLORS[1])
    right.set(xlabel="Optimizer step", ylabel=r"Log loss ($\mathtt{tabpfn\_val}$)")

    fig.tight_layout()
    save(fig, "training_curves")


if __name__ == "__main__":
    main()
