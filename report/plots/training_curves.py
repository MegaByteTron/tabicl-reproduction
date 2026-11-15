"""Figures from one training run: loss curves and ROC AUC over training.

    uv run python report/plots/training_curves.py <run folder> [results CSVs]

The run folder is the one holding metrics.jsonl and eval_datasets.jsonl
(outputs/runs/<date>/<time>/). Writes report/figures/training_curves.pdf and,
with results CSVs (e.g. "results/*.csv"), roc_auc_over_training.pdf.
make_figures.py makes these together with all other figures.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from style import (
    COLORS,
    FIGURES_DIR,
    INK_MUTED,
    TEXT_WIDTH_IN,
    is_ours,
    model_color,
    model_label,
    model_order,
    save,
    use_style,
)

from tabicl_reproduction.results import load_results
from tabicl_reproduction.score import N_ORDERS


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """metrics.jsonl (one row per logged step) and eval_datasets.jsonl
    (one row per eval step and tabpfn_val dataset) of a run."""
    metrics = pd.read_json(run_dir / "metrics.jsonl", lines=True)
    evals = pd.read_json(run_dir / "eval_datasets.jsonl", lines=True)
    return metrics, evals


def plot_training_curves(metrics: pd.DataFrame, outfile: Path, smooth: int = 20):
    """Training loss (prior) and log loss on tabpfn_val over the steps.

    The dotted line marks the step with the lowest validation log loss
    (saved as best.pt). Validation loss rising while the training loss
    keeps falling would mean overfitting.
    """
    train = metrics.dropna(subset=["train/loss"])
    evals = metrics.dropna(subset=["eval/log_loss"])
    best_step = evals.loc[evals["eval/log_loss"].idxmin(), "step"]

    use_style()
    # sharex: both panels use the same step range, so they line up.
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 1.9), sharex=True)

    # Raw values are noisy (one window of tables each); show them faintly
    # behind a moving average.
    left.plot(train["step"], train["train/loss"], color=COLORS[0], alpha=0.2)
    smoothed = train["train/loss"].rolling(smooth, min_periods=1).mean()
    left.plot(train["step"], smoothed, color=COLORS[0], label="moving average")
    left.set(xlabel="Optimizer step", ylabel="Training loss (prior)")
    left.legend()

    right.plot(evals["step"], evals["eval/log_loss"], color=COLORS[1])
    right.axvline(best_step, color=INK_MUTED, linestyle=":", label="best.pt")
    # The first evals are far higher; zoom in on the rest, where overfitting
    # would show.
    late = evals.loc[evals["step"] >= 0.05 * evals["step"].max(), "eval/log_loss"]
    margin = 0.1 * (late.max() - late.min())
    right.set(
        xlabel="Optimizer step",
        ylabel=r"Log loss ($\mathtt{tabpfn\_val}$)",
        ylim=(late.min() - margin, late.max() + margin),
    )
    right.legend(loc="upper right")

    fig.tight_layout()
    save(fig, outfile)


def plot_auc_over_training(evals: pd.DataFrame, results: pd.DataFrame, outfile: Path):
    """Our mean ROC AUC on tabpfn_val per eval step, baselines as lines.

    As in nanoTabPFN's plot_runs. evals comes from eval_datasets.jsonl
    (split 0); the baselines are averaged over the same tables of split 0
    from the results CSVs. Our scored checkpoint from the CSVs (averaged over
    column orders) is drawn as a single point, if present.
    """
    curve = evals.groupby("step")["roc_auc"].mean()
    ids = {str(i) for i in evals["dataset_id"].unique()}
    same = results[
        (results["source"] == "tabpfn_val")
        & (results["split_seed"] == 0)
        & (results["status"] == "ok")
        & results["table_id"].isin(ids)
    ]

    use_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.2))
    ax.plot(curve.index, curve.to_numpy(), color=COLORS[0], label="Ours (training)")
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5))]
    baselines = [m for m in model_order(same["model"]) if not is_ours(m)]
    means = [same.loc[same["model"] == m, "roc_auc"].mean() for m in baselines]
    for i, (model, mean) in enumerate(zip(baselines, means, strict=True)):
        ax.axhline(
            mean,
            color=model_color(model),
            linestyle=linestyles[i % len(linestyles)],
            label=model_label(model),
        )
    for model in [m for m in model_order(same["model"]) if is_ours(m)]:
        rows = same[same["model"] == model]
        step = rows["checkpoint_step"].iloc[0]
        ax.plot(
            step,
            rows["roc_auc"].mean(),
            marker="*",
            markersize=8,
            linestyle="",
            color=model_color(model),
            label=f"{model_label(model)} (scored, {N_ORDERS} column orders)",
        )
    # The first evals are near chance; zoom in on the rest and the baselines.
    late = curve[curve.index >= 0.05 * curve.index.max()]
    ax.set(
        xlabel="Optimizer step",
        ylabel=r"Mean ROC AUC ($\mathtt{tabpfn\_val}$)",
        ylim=(min([late.min(), *means]) - 0.02, None),
    )
    ax.legend(loc="lower right", ncols=2)
    fig.tight_layout()
    save(fig, outfile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("results", nargs="*", help="results CSVs (globs allowed)")
    parser.add_argument("--smooth", type=int, default=20, help="rows to average")
    args = parser.parse_args()

    metrics, evals = load_run(args.run_dir)
    plot_training_curves(metrics, FIGURES_DIR / "training_curves.pdf", args.smooth)
    if args.results:
        results = load_results(args.results)
        plot_auc_over_training(
            evals, results, FIGURES_DIR / "roc_auc_over_training.pdf"
        )


if __name__ == "__main__":
    main()
