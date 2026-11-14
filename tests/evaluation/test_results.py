import math
from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import confusion_matrix

from tabicl_reproduction.data import TableSample
from tabicl_reproduction.evaluation import compute_metrics, score_tables
from tabicl_reproduction.evaluation.core import table_info
from tabicl_reproduction.evaluation.metrics import (
    expected_calibration_error,
    roc_auc,
)
from tabicl_reproduction.evaluation.results import (
    COLUMNS,
    ResultsWriter,
    load_predictions,
    load_results,
    summarize,
)
from tabicl_reproduction.model import TabICLClassifier

# ----- metrics -----


def test_hand_computed_example():
    y = np.array([0, 0, 0, 1, -1])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
    m = compute_metrics(y, proba)
    assert m["accuracy"] == 0.75  # row 4 wrong, row 5 (-1) left out
    assert m["balanced_accuracy"] == 0.5  # class 0: 3/3, class 1: 0/1
    assert m["log_loss"] == pytest.approx(-np.log([0.9, 0.8, 0.7, 0.4]).mean())
    assert m["roc_auc"] == 1.0  # the class-1 row is ranked above all others
    assert m["ece"] == pytest.approx((0.1 + 0.2 + 0.3 + 0.6) / 4)


def test_minus_one_rows_are_ignored():
    y = np.array([0, 1, -1, 1])
    proba = np.array([[0.7, 0.3], [0.4, 0.6], [0.0, 1.0], [0.2, 0.8]])
    assert compute_metrics(y, proba) == compute_metrics(y[[0, 1, 3]], proba[[0, 1, 3]])


def test_roc_auc_is_nan_with_one_class():
    m = compute_metrics(np.array([1, 1]), np.array([[0.3, 0.7], [0.6, 0.4]]))
    assert math.isnan(m["roc_auc"])
    assert m["accuracy"] == 0.5


def test_roc_auc_handles_binary_and_multiclass():
    probs = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.1, 0.1, 0.8]])
    assert roc_auc(probs, np.array([0, 1, 2])) == 1.0
    assert roc_auc(probs[:2], np.array([0, 1])) == 1.0


def test_ece_is_zero_when_confidence_matches_accuracy():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, np.array([0, 1])) == 0.0
    assert expected_calibration_error(probs, np.array([1, 0])) == 1.0


def test_zero_probability_on_present_classes():
    # Test rows hold classes 0 and 1 only; the last row puts all mass on 2,
    # as k-NN or a random forest can. This must not crash.
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.6, 0.4, 0], [0.2, 0.8, 0], [0.8, 0.2, 0], [0, 0, 1.0]])
    m = compute_metrics(y, proba)
    assert all(math.isfinite(v) for v in m.values())


def test_float_labels_work():
    proba = np.array([[0.7, 0.3], [0.4, 0.6]])
    assert compute_metrics(torch.tensor([0.0, 1.0]), proba)["accuracy"] == 1.0


# ----- results CSV and predictions -----


def make_table(dataset_id=31, y_test=(0, 1, 2, 1, -1, 0), names=None, split=0):
    """3 classes in training; a test label of -1 is a class unseen in training."""
    y_train = torch.tensor([0, 1, 2, 0, 1, 2])
    y_test = torch.tensor(y_test)
    return TableSample(
        torch.randn(len(y_train), 4),
        y_train,
        torch.randn(len(y_test), 4),
        y_test,
        dataset_id=dataset_id,
        split=split,
        class_names=names,
    )


def test_round_trip(tmp_path, model):
    path = tmp_path / "results" / "ours.csv"
    writer = ResultsWriter(path, model="ours", checkpoint_step=9500)
    writer.git_commit = "1234567"  # all digits: must still load as text
    tables = [make_table(31, split=1), make_table(1, y_test=(1, 1, -1))]
    one_class = make_table(40)
    one_class.y_train[:] = 0
    clf = TabICLClassifier(model, n_orders=1)

    rows = score_tables(
        clf, tables[:1], on_row=partial(writer.add, source="tabpfn_val")
    )
    save_prior = partial(writer.add, source="prior_val", data_version="v2")
    rows += score_tables(clf, tables[1:], on_row=save_prior)
    rows += score_tables(
        clf, [one_class], on_row=partial(writer.add, source="tabpfn_val")
    )

    df = load_results(path)
    assert list(df.columns) == list(COLUMNS)
    assert list(df["table_id"]) == ["31", "prior_0001", "40"]
    assert list(df["status"]) == ["ok", "ok", "skipped"]
    ok = df.iloc[0]
    for name in compute_metrics(tables[0].y_test, np.full((6, 3), 1 / 3)):
        assert ok[name] == rows[0][name], name  # exactly, not approximately
    assert (ok["split"], ok["n_train"], ok["n_test"], ok["n_classes"]) == (1, 6, 5, 3)
    assert ok["checkpoint_step"] == 9500 and ok["git_commit"] == "1234567"
    assert ok["fit_s"] + ok["predict_s"] == pytest.approx(ok["time_s"])
    assert math.isnan(df.iloc[1]["roc_auc"])  # one class in the test rows
    assert df.iloc[1]["data_version"] == "v2"
    assert df.iloc[2]["error"] == "fewer than 2 classes in training"
    assert math.isnan(df.iloc[2]["accuracy"])


def test_predictions_give_a_confusion_matrix(tmp_path, model):
    writer = ResultsWriter(
        tmp_path / "ours.csv", model="ours", predictions_dir=tmp_path / "pred"
    )
    table = make_table(31, names=("cat", "dog", "fish"), split=3)
    clf = TabICLClassifier(model, n_orders=1)
    score_tables(clf, [table], on_row=partial(writer.add, source="tabpfn_val"))

    p = load_predictions(tmp_path / "pred" / "tabpfn_val" / "31_split3.npz")
    assert list(p["class_names"]) == ["cat", "dog", "fish"]
    assert list(p["y_true"]) == table.y_test.tolist()
    assert p["proba"].shape == (6, 3)
    ok = p["y_true"] >= 0
    matrix = confusion_matrix(
        p["y_true"][ok], p["proba"][ok].argmax(1), labels=[0, 1, 2]
    )
    assert matrix.sum() == 5


def skipped_row():
    """A row as score_table makes it for a skipped table."""
    return table_info(make_table()) | {"status": "skipped", "error": "no"}


def test_skipped_tables_have_no_predictions(tmp_path):
    writer = ResultsWriter(
        tmp_path / "x.csv", model="x", predictions_dir=tmp_path / "p"
    )
    writer.add(
        skipped_row(),
        None,
        make_table(),
        source="tabpfn_val",
    )
    assert not (tmp_path / "p").exists()


def test_appends_to_existing_file(tmp_path):
    path = tmp_path / "rf.csv"
    for _ in range(2):  # e.g. a run that resumes after a time limit
        ResultsWriter(path, model="rf").add(skipped_row(), source="tabpfn_val")
    assert len(load_results(path)) == 2


def test_wrong_columns_are_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("model,accuracy,extra\nrf,0.5,1\n")
    with pytest.raises(ValueError, match="missing columns.*unknown columns.*extra"):
        load_results(bad)
    with pytest.raises(ValueError, match="other columns"):
        ResultsWriter(bad, model="rf")


def test_no_matching_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "*.csv")
    with pytest.raises(FileNotFoundError):
        load_results([])


def test_file_without_rows_keeps_number_types(tmp_path, model):
    ResultsWriter(tmp_path / "empty.csv", model="empty")  # header only
    writer = ResultsWriter(tmp_path / "ours.csv", model="ours")
    score_tables(
        TabICLClassifier(model, n_orders=1),
        [make_table()],
        on_row=partial(writer.add, source="tabpfn_val"),
    )
    df = load_results(tmp_path / "*.csv")
    assert df["n_train"].dtype == "int64" and df["accuracy"].dtype == "float64"


# ----- summary -----


def row(
    model, table_id, roc_auc=0.5, step=math.nan, source="tabpfn_val", ok=True, split=0
):
    return dict.fromkeys(COLUMNS, math.nan) | {
        "model": model,
        "source": source,
        "table_id": table_id,
        "split": split,
        "checkpoint_step": step,
        "roc_auc": roc_auc if ok else math.nan,
        "log_loss": 1 - roc_auc,
        "status": "ok" if ok else "skipped",
    }


def n_tables(rows):
    summary = summarize(pd.DataFrame(rows))
    keys = summary["model"] + "/" + summary["source"]
    return dict(zip(keys, summary["n_tables"], strict=True))


@pytest.mark.parametrize("skipper", ["rf", "ours"])
def test_summary_counts_only_tables_every_model_scored(skipper):
    rows = [
        row("rf", "31", 0.8),
        row("rf", "40", 0.6, ok=skipper != "rf"),
        row("ours", "31", 0.9, step=1000),
        row("ours", "40", 0.7, step=1000, ok=skipper != "ours"),
    ]
    assert n_tables(rows) == {"ours/tabpfn_val": 1, "rf/tabpfn_val": 1}
    summary = summarize(pd.DataFrame(rows)).set_index("model")
    assert summary.loc["rf", "roc_auc_mean"] == 0.8  # table 40 left out
    assert summary.loc["ours", "roc_auc_mean"] == 0.9


def test_summary_per_source():
    # Our model only has OpenML results (e.g. a test run): prior rows stay.
    rows = [
        row("rf", "prior_0000", source="prior_val"),
        row("rf", "31"),
        row("ours", "31", step=1000),
    ]
    assert n_tables(rows) == {
        "rf/prior_val": 1,
        "rf/tabpfn_val": 1,
        "ours/tabpfn_val": 1,
    }


def test_summary_ignores_a_run_without_any_scored_table():
    rows = [row("rf", "31"), row("ours", "31", step=1000, ok=False)]
    assert n_tables(rows) == {"rf/tabpfn_val": 1}


def test_summary_with_checkpoints_and_nan():
    rows = [
        row("rf", "31", 0.8),
        row("rf", "40", math.nan),  # e.g. one class in the test rows
        row("ours", "31", 0.9, step=1000),
        row("ours", "40", 0.7, step=1000),
        row("ours", "31", 0.95, step=2000),
        row("ours", "40", 0.75, step=2000),
    ]
    summary = summarize(pd.DataFrame(rows))
    assert list(summary["n_tables"]) == [2, 2, 2]  # each checkpoint on its own
    assert list(summary["roc_auc_mean"]) == pytest.approx([0.8, 0.85, 0.8])


def test_summary_averages_folds_before_tables():
    # Table 31 has 3 folds, table 40 one: each table still counts once.
    rows = [
        row("rf", "31", 0.6, split=0),
        row("rf", "31", 0.7, split=1),
        row("rf", "31", 0.8, split=2),
        row("rf", "40", 0.9),
    ]
    summary = summarize(pd.DataFrame(rows)).iloc[0]
    assert summary["n_tables"] == 2
    assert summary["roc_auc_mean"] == pytest.approx((0.7 + 0.9) / 2)


def test_summary_mean_rank():
    rows = [
        row("rf", "31", 0.8),
        row("rf", "40", 0.6),
        row("ours", "31", 0.9, step=1),  # better on 31
        row("ours", "40", 0.5, step=1),  # worse on 40
        row("knn", "31", 0.1),
        row("knn", "40", 0.1),
    ]
    summary = summarize(pd.DataFrame(rows)).set_index("model")
    assert summary.loc["ours", "roc_auc_mean_rank"] == 1.5  # ranks 1 and 2
    assert summary.loc["rf", "roc_auc_mean_rank"] == 1.5
    assert summary.loc["knn", "roc_auc_mean_rank"] == 3
    # log_loss: lower is better, so the order of the ranks flips.
    assert summary.loc["knn", "log_loss_mean_rank"] == 3


def test_summary_refuses_duplicate_rows():
    rows = [row("rf", "31"), row("rf", "31")]
    with pytest.raises(ValueError, match="two rows for the same table"):
        summarize(pd.DataFrame(rows))
