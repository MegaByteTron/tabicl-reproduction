import math

import numpy as np
import pytest
import torch

from tabicl_reproduction.data import TableSample, to_model_inputs
from tabicl_reproduction.results import (
    COLUMNS,
    ResultsWriter,
    compute_metrics,
    load_results,
)
from tabicl_reproduction.training.evaluator import Evaluator


def make_table(y_test=(0, 1, 2, 1, -1, 0)) -> TableSample:
    """3 classes in training; a test label of -1 is a class unseen in training."""
    y_train = torch.tensor([0, 1, 2, 0, 1, 2])
    y_test = torch.tensor(y_test)
    return TableSample(
        torch.randn(len(y_train), 4), y_train, torch.randn(len(y_test), 4), y_test
    )


@torch.no_grad()
def test_same_values_as_evaluator(model):
    table = make_table()
    expected = Evaluator(model, "cpu", tables=[table]).evaluate()["datasets"][0]

    # How our model passes proba: softmax over the training classes,
    # for all test rows (-1 rows included).
    logits = model(*to_model_inputs(table))[0]
    proba = torch.softmax(logits[:, :3], dim=-1)
    metrics = compute_metrics(table.y_test, proba)
    # Equal for ordinary predictions; log_loss only differs past the 1e-15 clip.
    for name, value in metrics.items():
        assert value == pytest.approx(expected[name], rel=1e-6), name


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


def test_zero_probability_on_present_classes():
    # Test rows hold classes 0 and 1 only; the last row puts all mass on 2,
    # as k-NN or a random forest can. This must not crash.
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.6, 0.4, 0], [0.2, 0.8, 0], [0.8, 0.2, 0], [0, 0, 1.0]])
    m = compute_metrics(y, proba)
    assert all(math.isfinite(v) for v in m.values())


def test_round_trip(tmp_path):
    path = tmp_path / "results" / "rf.csv"
    writer = ResultsWriter(path, model="rf")
    writer.git_commit = "1234567"  # all digits: must still load as text
    table = make_table()
    one_class = make_table(y_test=(1, 1, -1))
    proba = torch.softmax(torch.randn(6, 3), dim=-1)
    proba_one = torch.softmax(torch.randn(3, 3), dim=-1)

    writer.add(
        table, proba, source="tabpfn_val", table_id="31", split_seed=0, time_s=0.5
    )
    writer.add(
        one_class,
        proba_one,
        source="prior_val",
        table_id="prior_0001",
        split_seed=0,
        time_s=0.1,
        data_version="v1",
    )
    # A reason with a comma, quotes and a newline must survive the CSV.
    reason = 'ValueError: bad, "odd"\nsecond line'
    writer.skip(table, reason, source="tabpfn_val", table_id="40", split_seed=1)

    df = load_results(path)
    assert list(df.columns) == list(COLUMNS)
    assert list(df["status"]) == ["ok", "ok", "skipped"]
    assert list(df["table_id"]) == ["31", "prior_0001", "40"]

    ok = df.iloc[0]
    for name, value in compute_metrics(table.y_test, proba).items():
        assert ok[name] == value, name  # exactly, not approximately
    assert (ok["n_train"], ok["n_test"], ok["n_features"], ok["n_classes"]) == (
        6,
        5,
        4,
        3,
    )
    assert ok["time_s"] == 0.5
    assert math.isnan(ok["checkpoint_step"]) and ok["git_commit"] == "1234567"

    assert math.isnan(df.iloc[1]["roc_auc"])  # one class in the test rows
    assert df.iloc[1]["data_version"] == "v1"

    skipped = df.iloc[2]
    assert skipped["error"] == reason
    assert all(math.isnan(skipped[m]) for m in ("accuracy", "roc_auc", "time_s"))


def test_appends_to_existing_file(tmp_path):
    path = tmp_path / "rf.csv"
    table = make_table()
    for _ in range(2):  # e.g. a run that resumes after a time limit
        ResultsWriter(path, model="rf").skip(
            table, "x", source="tabpfn_val", table_id="31", split_seed=0
        )
    assert len(load_results(path)) == 2


def test_table_without_scorable_rows_is_skipped(tmp_path):
    path = tmp_path / "rf.csv"
    table = make_table(y_test=(-1, -1))
    proba = np.full((2, 3), 1 / 3)
    ResultsWriter(path, model="rf").add(
        table, proba, source="tabpfn_val", table_id="31", split_seed=0, time_s=0.1
    )
    row = load_results(path).iloc[0]
    assert row["status"] == "skipped" and row["n_test"] == 0


def test_wrong_proba_shape_raises(tmp_path):
    writer = ResultsWriter(tmp_path / "rf.csv", model="rf")
    with pytest.raises(ValueError, match="proba must have shape"):
        writer.add(
            make_table(),
            np.full((6, 2), 0.5),
            source="tabpfn_val",
            table_id="31",
            split_seed=0,
            time_s=0.1,
        )


def test_wrong_columns_are_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("model,accuracy,extra\nrf,0.5,1\n")
    with pytest.raises(ValueError, match="missing columns.*unknown columns.*extra"):
        load_results(bad)
    with pytest.raises(ValueError, match="other columns"):
        ResultsWriter(bad, model="rf")


def test_two_models_load_together(tmp_path):
    table = make_table()
    proba = torch.softmax(torch.randn(6, 3), dim=-1)
    for model in ("rf", "knn"):
        ResultsWriter(tmp_path / f"{model}.csv", model=model).add(
            table, proba, source="tabpfn_val", table_id="31", split_seed=0, time_s=0.1
        )
    df = load_results(tmp_path / "*.csv")
    assert sorted(df["model"]) == ["knn", "rf"]


def test_no_matching_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "*.csv")
    with pytest.raises(FileNotFoundError):
        load_results([])


def test_file_without_rows_keeps_number_types(tmp_path):
    ResultsWriter(tmp_path / "empty.csv", model="empty")  # header only
    ResultsWriter(tmp_path / "rf.csv", model="rf").add(
        make_table(),
        np.full((6, 3), 1 / 3),
        source="tabpfn_val",
        table_id="31",
        split_seed=0,
        time_s=0.1,
    )
    df = load_results(tmp_path / "*.csv")
    assert df["n_train"].dtype == "int64" and df["accuracy"].dtype == "float64"


def test_float_labels_work():
    proba = np.array([[0.7, 0.3], [0.4, 0.6]])
    assert compute_metrics(torch.tensor([0.0, 1.0]), proba)["accuracy"] == 1.0
