import math

import numpy as np
import pandas as pd
import pytest
import torch

from tabicl_reproduction.baselines import BASELINES, run_all, run_baselines, summarize
from tabicl_reproduction.data import TableSample
from tabicl_reproduction.prior_val import save_prior_tables
from tabicl_reproduction.results import COLUMNS, load_results


def make_table(dataset_id, n_classes=3, n_train=30, n_test=10):
    """Easy table: the class is the rounded first feature, plus noise."""
    y_train = torch.arange(n_train) % n_classes
    y_test = torch.arange(n_test) % n_classes
    X_train = torch.randn(n_train, 4) * 0.1
    X_test = torch.randn(n_test, 4) * 0.1
    X_train[:, 0] += y_train
    X_test[:, 0] += y_test
    return TableSample(X_train, y_train, X_test, y_test, dataset_id=dataset_id)


def test_every_baseline_on_every_table(tmp_path):
    tables = [make_table(0), make_table(1, n_classes=1), make_table(2, n_classes=2)]
    run_baselines(tables, tmp_path, source="prior_val", data_version="v1")

    df = load_results(tmp_path / "*.csv")
    assert sorted(df["model"].unique()) == sorted(BASELINES)
    for _, rows in df.groupby("model"):
        assert list(rows["table_id"]) == ["prior_0000", "prior_0001", "prior_0002"]
        assert list(rows["status"]) == ["ok", "skipped", "ok"]
    skipped = df[df["status"] == "skipped"]
    assert set(skipped["error"]) == {"fewer than 2 classes in training"}
    assert (df["data_version"] == "v1").all()


def test_openml_table_ids_and_split_seed(tmp_path):
    run_baselines([make_table(31)], tmp_path, source="tabpfn_val", split_seed=1)
    df = load_results(tmp_path / "*.csv")
    assert set(df["table_id"]) == {"31"} and set(df["split_seed"]) == {1}


class Broken:
    def fit(self, X, y):
        raise RuntimeError("cannot fit")


class WrongShape:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.ones((len(X), 1))  # one column too few


def test_failing_model_is_skipped(tmp_path):
    run_baselines(
        [make_table(0)], tmp_path, source="prior_val", baselines={"broken": Broken}
    )
    row = load_results(tmp_path / "broken.csv").iloc[0]
    assert row["status"] == "skipped" and row["error"] == "RuntimeError: cannot fit"


def test_bug_in_our_code_is_not_hidden(tmp_path):
    with pytest.raises(ValueError, match="proba must have shape"):
        run_baselines(
            [make_table(0)], tmp_path, source="prior_val", baselines={"w": WrongShape}
        )


def row(model, table_id, roc_auc=0.5, step=math.nan, source="tabpfn_val", ok=True):
    return dict.fromkeys(COLUMNS, math.nan) | {
        "model": model,
        "source": source,
        "table_id": table_id,
        "split_seed": 0,
        "checkpoint_step": step,
        "roc_auc": roc_auc if ok else math.nan,
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


def test_summary_with_baselines_only():
    # No checkpoint at all: one baseline skipped table 40.
    rows = [
        row("rf", "31"),
        row("rf", "40", ok=False),
        row("knn", "31"),
        row("knn", "40"),
    ]
    assert n_tables(rows) == {"knn/tabpfn_val": 1, "rf/tabpfn_val": 1}


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


def test_summary_refuses_duplicate_rows():
    rows = [row("rf", "31"), row("rf", "31")]
    with pytest.raises(ValueError, match="two rows for the same table"):
        summarize(pd.DataFrame(rows))


def test_full_run_refuses_a_small_test_set(tmp_path, monkeypatch):
    monkeypatch.setenv("TABICL_DATASETS_DIR", str(tmp_path))
    save_prior_tables(tmp_path / "prior_val", n_tables=2)
    with pytest.raises(SystemExit, match="has 2 tables, the full run needs 1600"):
        run_all(tmp_path / "results")
