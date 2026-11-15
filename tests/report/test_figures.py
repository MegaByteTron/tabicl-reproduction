import numpy as np
import pytest
from comparison import (
    best_baseline,
    per_table,
    plot_by_category,
    plot_differences,
    plot_mean_rank,
    plot_per_dataset,
    table_info,
)
from confusion import ConfusionCase, plot_confusion_matrices
from make_figures import win_and_loss
from training_curves import plot_auc_over_training, plot_training_curves


def test_per_table_averages_splits_and_keeps_common_tables(results):
    # knn skipped one table: no model may count it.
    results.loc[
        (results["model"] == "knn") & (results["table_id"] == "459"), "status"
    ] = "skipped"
    scores = per_table(results, "tabpfn_val")
    assert list(scores.columns) == ["tabicl_best", "random_forest", "knn"]
    assert len(scores) == 5 and "459" not in scores.index
    rows = results[(results["model"] == "knn") & (results["table_id"] == "1037")]
    assert scores.loc["1037", "knn"] == pytest.approx(rows["roc_auc"].mean())


def test_best_baseline_is_never_ours(results):
    scores = per_table(results, "tabpfn_val")
    scores["tabicl_best"] = 1.0
    assert best_baseline(scores) in ("random_forest", "knn")


@pytest.mark.parametrize(
    "plot", [plot_differences, plot_mean_rank, plot_per_dataset, plot_by_category]
)
def test_comparison_plots_write_pdf(tmp_path, results, plot):
    out = tmp_path / "figure.pdf"
    plot(results, out)
    assert out.stat().st_size > 0


def test_run_plots_write_pdf(tmp_path, results, run_logs):
    metrics, evals = run_logs
    plot_training_curves(metrics, tmp_path / "loss.pdf")
    plot_auc_over_training(evals, results, tmp_path / "auc.pdf")
    assert (tmp_path / "loss.pdf").stat().st_size > 0
    assert (tmp_path / "auc.pdf").stat().st_size > 0


def test_confusion_matrices_write_pdf(tmp_path):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 50)
    cases = [
        ConfusionCase("win", y_true, {"tabicl_best": y_true, "knn": y_true[::-1]}),
        # A predicted class missing from y_true must not break the matrix.
        ConfusionCase("loss", np.zeros(5, int), {"knn": np.array([0, 1, 2, 3, 0])}),
    ]
    out = tmp_path / "confusion.pdf"
    plot_confusion_matrices(cases, out)
    assert out.stat().st_size > 0


def test_win_and_loss_picks_extremes(results):
    scores = per_table(results, "tabpfn_val", "balanced_accuracy")
    info = table_info(results, "tabpfn_val")
    tables = win_and_loss(scores, info, "tabicl_best", "random_forest")
    diff = scores["tabicl_best"] - scores["random_forest"]
    readable = diff[info.loc[diff.index, "n_classes"].between(3, 6)]
    assert tables == {readable.idxmax(): "ours wins", readable.idxmin(): "ours loses"}


def test_results_table_bolds_best_per_source(tmp_path, results):
    from tables import results_table

    results.loc[results["model"] == "knn", "roc_auc"] = 1.0
    results.loc[results["model"] == "knn", "ece"] = 0.0
    text = results_table(results, tmp_path / "results.tex").read_text()
    knn_rows = [line for line in text.splitlines() if line.startswith("$k$-NN")]
    assert len(knn_rows) == 2  # one per source
    for row in knn_rows:
        cells = row.split(" & ")
        assert cells[1] == r"\textbf{1.000}" and r"\textbf{0.000}" in cells[5]
    assert text.count(r"\midrule") == 2
