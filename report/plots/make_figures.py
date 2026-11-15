"""Make every report figure from a training run and the results CSVs.

    uv run python report/plots/make_figures.py <run folder>
    uv run python report/plots/make_figures.py <run folder> --results "results/*.csv"

The run folder holds metrics.jsonl, eval_datasets.jsonl, .hydra/ and the
weights (outputs/runs/<date>/<time>/). Writes PDFs into report/figures/
(or --out) and the results table into report/tables/. The confusion matrices predict a few OpenML tables again, which
takes a few seconds on the CPU.
"""

import argparse
from pathlib import Path

import pandas as pd
from comparison import (
    best_baseline,
    dataset_label,
    per_table,
    plot_by_category,
    plot_differences,
    plot_per_dataset,
    table_info,
)
from confusion import ConfusionCase, plot_confusion_matrices
from hydra_zen import instantiate, load_from_yaml
from style import FIGURES_DIR, is_ours, model_order
from tables import results_table
from training_curves import load_run, plot_training

from tabicl_reproduction.baselines import BASELINES
from tabicl_reproduction.data import load_data
from tabicl_reproduction.model.classifier import NanoTabICLv2Classifier
from tabicl_reproduction.results import load_results
from tabicl_reproduction.training.checkpoint import load_model


def win_and_loss(
    scores: pd.DataFrame, info: pd.DataFrame, ours: str, baseline: str
) -> dict[str, str]:
    """The OpenML tables where we beat the baseline most and lose most.

    scores: balanced accuracy per table (per_table), since a confusion matrix
    shows hard predictions. Prefers tables with 3 to 6 classes (more to see
    than 2x2, still readable) and at least 100 test rows (so the fractions
    are not from a handful of rows). Returns {table_id: "win" or "loss"}.
    """
    diff = scores[ours] - scores[baseline]
    info = info.loc[diff.index]
    readable = info["n_classes"].between(3, 6) & (info["n_test"] >= 100)
    if readable.sum() >= 2:
        diff = diff[readable]
    return {diff.idxmax(): "ours wins", diff.idxmin(): "ours loses"}


def confusion_cases(
    run_dir: Path, ours: str, baseline: str, tables: dict[str, str]
) -> list[ConfusionCase]:
    """Predict the test rows of split 0 of these tabpfn_val tables again.

    tables: {table_id: note for the title}. Ours is one forward pass with
    NanoTabICLv2Classifier (the CSVs average 8 column orders).
    """
    config = load_from_yaml(run_dir / ".hydra" / "config.yaml")
    model = instantiate(config.model)
    load_model(model, run_dir, ours.removeprefix("tabicl_"))  # tabicl_best: best.pt
    classifier = NanoTabICLv2Classifier(model.eval())

    found = {}
    for table in load_data("tabpfn_val", split_seed=0):
        table_id = str(table.dataset_id)
        if table_id not in tables:
            continue
        X_train, y_train, X_test, y_test = (t.numpy() for t in table)
        keep = y_test >= 0  # -1: a class not in the training rows
        X_test, y_test = X_test[keep], y_test[keep]
        y_pred = {
            ours: classifier.fit(X_train, y_train).predict(X_test),
            baseline: BASELINES[baseline]().fit(X_train, y_train).predict(X_test),
        }
        title = f"{dataset_label(table_id)} ({tables[table_id]})"
        found[table_id] = ConfusionCase(title, y_test, y_pred)
    return [found[t] for t in tables]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="training run folder")
    parser.add_argument("--results", nargs="+", default=["results/*.csv"])
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    out: Path = args.out

    metrics, evals = load_run(args.run_dir)
    results = load_results(args.results)
    plot_training(metrics, evals, results, out / "training.pdf")
    plot_differences(results, out / "roc_auc_differences.pdf")
    plot_by_category(results, out / "roc_auc_by_category.pdf")
    results_table(results, out.parent / "tables" / "results.tex")

    ours = [m for m in model_order(results["model"]) if is_ours(m)]
    if not ours:
        print("No results of our model (tabicl_*.csv): skipped the per-dataset")
        print("comparison and the confusion matrices.")
        return
    plot_per_dataset(results, out / "per_dataset.pdf", ours=ours[0])

    # Against the best baseline by ROC AUC, as in the per-dataset figure.
    baseline = best_baseline(per_table(results, "tabpfn_val"))
    scores = per_table(results, "tabpfn_val", "balanced_accuracy")
    info = table_info(results, "tabpfn_val")
    tables = win_and_loss(scores, info, ours[0], baseline)
    cases = confusion_cases(args.run_dir, ours[0], baseline, tables)
    plot_confusion_matrices(cases, out / "confusion_matrices.pdf")


if __name__ == "__main__":
    main()
