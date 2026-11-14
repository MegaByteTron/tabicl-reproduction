import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier

from tabicl_reproduction.data import TableSample
from tabicl_reproduction.evaluation import compute_metrics, score_table, score_tables
from tabicl_reproduction.model import TabICLClassifier


def make_table(dataset_id=0, n_classes=3, n_train=30, n_test=10, split=0):
    """Easy table: the class is the rounded first feature, plus noise."""
    y_train = torch.arange(n_train) % n_classes
    y_test = torch.arange(n_test) % n_classes
    X_train = torch.randn(n_train, 4) * 0.1
    X_test = torch.randn(n_test, 4) * 0.1
    X_train[:, 0] += y_train
    X_test[:, 0] += y_test
    return TableSample(
        X_train, y_train, X_test, y_test, dataset_id=dataset_id, split=split
    )


def test_baseline_and_our_model_give_the_same_row(model):
    table = make_table(dataset_id=31, split=2)
    rows = {}
    for name, estimator in {
        "rf": RandomForestClassifier(random_state=0),
        "ours": TabICLClassifier(model, n_orders=1),
    }.items():
        row, proba = score_table(estimator, table)
        assert proba is not None and proba.shape == (10, 3)
        assert row["status"] == "ok" and row["error"] is None
        assert {k: row[k] for k in compute_metrics(table.y_test, proba)} == (
            compute_metrics(table.y_test, proba)
        )
        rows[name] = row
    assert rows["rf"].keys() == rows["ours"].keys()
    info = {k: rows["rf"][k] for k in ("dataset_id", "split", "n_train", "n_classes")}
    assert info == {"dataset_id": 31, "split": 2, "n_train": 30, "n_classes": 3}
    assert rows["rf"]["accuracy"] == 1.0  # the easy table


def test_tables_that_cannot_be_scored_are_skipped(model):
    one_class = make_table(n_classes=1)
    no_scorable_row = TableSample(*make_table(), dataset_id=0)
    no_scorable_row.y_test[:] = -1
    too_many = make_table(n_classes=4)
    clf = TabICLClassifier(model, max_classes=3)
    rows = score_tables(clf, [one_class, no_scorable_row, too_many])
    assert [r["status"] for r in rows] == ["skipped"] * 3
    assert [r["error"] for r in rows] == [
        "fewer than 2 classes in training",
        "no test row with a class seen in training",
        "4 classes, the model needs 2 to 3",
    ]


class Broken:
    def fit(self, X, y):
        raise RuntimeError("cannot fit")


class WrongShape:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.ones((len(X), 1))  # one column too few


def test_errors_raise_unless_skipped():
    with pytest.raises(RuntimeError, match="cannot fit"):
        score_table(Broken(), make_table())
    row, proba = score_table(Broken(), make_table(), skip_errors=True)
    assert row["error"] == "RuntimeError: cannot fit" and proba is None


def test_bug_in_our_code_is_not_hidden():
    with pytest.raises(ValueError, match="proba must have shape"):
        score_table(WrongShape(), make_table(), skip_errors=True)


def test_on_row_gets_every_result_and_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = []
    tables = [make_table(0), make_table(1, n_classes=1)]
    rows = score_tables(
        RandomForestClassifier(random_state=0),
        tables,
        on_row=lambda row, proba, table: seen.append((row, proba, table)),
    )
    assert [s[0] for s in seen] == rows
    assert seen[0][1].shape == (10, 3) and seen[1][1] is None
    assert seen[1][2] is tables[1]
    assert list(tmp_path.iterdir()) == []  # scoring alone writes no file
