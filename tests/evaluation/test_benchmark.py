import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier

from tabicl_reproduction.data import TableSample
from tabicl_reproduction.data.prior_val import save_prior_tables
from tabicl_reproduction.evaluation import benchmark
from tabicl_reproduction.evaluation.benchmark import (
    Source,
    run_benchmark,
    score_table,
    score_tables,
    sources,
)
from tabicl_reproduction.evaluation.metrics import evaluate_metrics, metric_names
from tabicl_reproduction.evaluation.models import get_models
from tabicl_reproduction.evaluation.results import load_results
from tabicl_reproduction.model import TabICLClassifier


def make_table(dataset_id=0, n_classes=3, split=0, n_train=30, n_test=10):
    """Easy table: the class is the rounded first feature, plus noise."""
    y_train = torch.arange(n_train) % n_classes
    y_test = torch.arange(n_test) % n_classes
    X_train = torch.randn(n_train, 4) * 0.1
    X_test = torch.randn(n_test, 4) * 0.1
    X_train[:, 0] += y_train
    X_test[:, 0] += y_test
    return TableSample(X_train, y_train, X_test, y_test, dataset_id, split)


# ----- scoring one table -----


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
        metrics = evaluate_metrics(table.y_test, proba)
        assert {k: row[k] for k in metrics} == metrics
        rows[name] = row
    assert rows["rf"].keys() == rows["ours"].keys()
    info = {k: rows["rf"][k] for k in ("dataset_id", "split", "n_train", "n_classes")}
    assert info == {"dataset_id": 31, "split": 2, "n_train": 30, "n_classes": 3}
    assert rows["rf"]["accuracy"] == 1.0  # the easy table


class CountingForest(RandomForestClassifier):
    calls = 0

    def predict_proba(self, X):
        CountingForest.calls += 1
        return super().predict_proba(X)


def test_one_prediction_per_table_for_every_metric():
    CountingForest.calls = 0
    score_table(CountingForest(random_state=0), make_table())
    assert CountingForest.calls == 1
    assert len(metric_names()) > 1


def test_tables_that_cannot_be_scored_are_skipped(model):
    one_class = make_table(n_classes=1)
    no_scorable_row = make_table()
    no_scorable_row.y_test[:] = -1
    too_many = make_table(n_classes=4)
    rows = score_tables(
        TabICLClassifier(model, max_classes=3), [one_class, no_scorable_row, too_many]
    )
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


def test_scoring_writes_nothing(tmp_path, monkeypatch):
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
    assert list(tmp_path.iterdir()) == []


# ----- the benchmark -----


def fake_sources():
    openml = [make_table(31, split=0), make_table(31, split=1), make_table(40, 1)]
    prior = [make_table(0), make_table(1, n_classes=2)]
    return [
        Source("tabpfn_val", openml, len(openml)),
        Source("prior_val", prior, len(prior), data_version="v2"),
    ]


def test_every_model_on_every_table(tmp_path, model):
    models = dict(get_models()) | {
        "nano_tabicl_v2": TabICLClassifier(model, n_orders=1)
    }
    run_benchmark(models, fake_sources(), tmp_path)

    df = load_results(tmp_path / "*.csv")
    assert sorted(df["model"].unique()) == sorted(models)
    for _, rows in df.groupby("model"):
        assert list(rows["table_id"]) == ["31", "31", "40", "prior_0000", "prior_0001"]
        assert list(rows["split"]) == [0, 1, 0, 0, 0]
        assert list(rows["status"]) == ["ok", "ok", "skipped", "ok", "ok"]
    assert set(df[df["source"] == "prior_val"]["data_version"]) == {"v2"}
    files = sorted((tmp_path / "predictions" / "nano_tabicl_v2").rglob("*.npz"))
    assert [f"{f.parent.name}/{f.name}" for f in files] == [
        "prior_val/prior_0000_split0.npz",
        "prior_val/prior_0001_split0.npz",
        "tabpfn_val/31_split0.npz",
        "tabpfn_val/31_split1.npz",
    ]


def test_refuses_to_append_to_existing_results(tmp_path, model):
    models = {"nano_tabicl_v2": TabICLClassifier(model, n_orders=1)}
    run_benchmark(models, fake_sources(), tmp_path, save_predictions=False)
    assert not (tmp_path / "predictions").exists()
    with pytest.raises(SystemExit, match="already has results for"):
        run_benchmark(models, fake_sources(), tmp_path)


def test_command_writes_the_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "sources", lambda cv, limit, n: fake_sources())
    benchmark.main(["--out", str(tmp_path), "--models", "knn", "random_forest"])
    assert (tmp_path / "summary.txt").exists()
    assert "random_forest" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="needs a checkpoint"):
        benchmark.main(["--out", str(tmp_path / "x"), "--models", "nano_tabicl_v2"])


def test_full_run_refuses_a_small_prior_set(tmp_path, monkeypatch):
    monkeypatch.setenv("TABICL_DATASETS_DIR", str(tmp_path))
    save_prior_tables(tmp_path / "prior_val", n_tables=2)
    with pytest.raises(SystemExit, match="has 2 tables, the full run needs 1600"):
        sources(cv=None)
