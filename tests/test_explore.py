import math
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from tabicl_reproduction import explore
from tabicl_reproduction.data import TableSample
from tabicl_reproduction.explore import raw_stats, summarize, table_stats


def small_table(dataset_id=0):
    # Column 0 has one value in training; column 1 has mean 2.5 and std sqrt(1.25).
    X_train = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
    y_train = torch.tensor([0, 0, 0, 1])
    # Row 0 repeats a training row with its label; row 1 is far out and its
    # class (-1) never appears in training.
    X_test = torch.tensor([[0.0, 2.0], [0.0, 100.0]])
    y_test = torch.tensor([0, -1])
    return TableSample(X_train, y_train, X_test, y_test, dataset_id=dataset_id)


def test_table_stats():
    stats = table_stats(small_table())
    assert stats["n_train"] == 4 and stats["n_test"] == 2
    assert stats["n_features"] == 2 and stats["n_classes"] == 2
    assert stats["majority_share"] == 0.75
    assert stats["unseen_test_rows"] == 1
    assert stats["one_value_columns"] == 1
    assert stats["duplicate_test_rows"] == 1
    assert stats["duplicate_test_rows_same_label"] == 1
    assert stats["max_abs_z"] == pytest.approx((100 - 2.5) / math.sqrt(1.25))


def test_duplicate_with_another_label():
    X_train, y_train, X_test, _ = small_table()
    stats = table_stats(TableSample(X_train, y_train, X_test, torch.tensor([1, 0])))
    assert stats["duplicate_test_rows"] == 1
    assert stats["duplicate_test_rows_same_label"] == 0


def test_all_columns_constant():
    X_train, y_train, _, y_test = small_table()
    table = TableSample(X_train[:, :1], y_train, torch.zeros(2, 1), y_test)
    assert table_stats(table)["max_abs_z"] == 0.0


def test_raw_stats():
    X = pd.DataFrame({"color": ["red", None, "blue", "red"], "size": [1.0] * 4})
    assert raw_stats(X) == {"missing_share": 1 / 8, "categorical_columns": 1}


def test_explore_command(tmp_path, monkeypatch):
    # A fake OpenML dataset on disk, stored the way OpenMLDataset caches it.
    monkeypatch.setenv("TABICL_DATASETS_DIR", str(tmp_path / "data"))
    folder = tmp_path / "data" / "adaprior"
    folder.mkdir(parents=True)
    features = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", None]})
    features.to_parquet(folder / "features.parquet")
    target = pd.DataFrame({"target": ["yes", "no", None]})  # one unlabeled row
    target.to_parquet(folder / "target.parquet")

    def fake_result_tables(limit):
        yield {"source": "tabpfn_val", "split_seed": 0}, [small_table(1037)]
        yield {"source": "prior_val", "split_seed": 0}, [small_table(0)]

    monkeypatch.setattr(explore, "result_tables", fake_result_tables)
    out = tmp_path / "data_stats.csv"
    monkeypatch.setattr(sys, "argv", ["explore", "--out", str(out)])
    explore.main()

    df = pd.read_csv(out)
    assert df["table_id"].tolist() == ["1037", "prior_0000"]
    # Only the 2 labeled rows count: 1 of their 4 cells is missing.
    assert df["missing_share"].tolist()[0] == 0.25
    assert df["categorical_columns"].tolist()[0] == 1
    assert df[["missing_share", "categorical_columns"]].iloc[1].isna().all()

    summary = summarize(df)
    assert summary.loc[("prior_val", 0), "tables"] == 1
    assert summary.loc[("prior_val", 0), "duplicate_row_share"] == 0.5  # 1 of 2
    assert summary.loc[("prior_val", 0), "abs_z_over_10"] == 1
    assert summary.loc[("tabpfn_val", 0), "missing_values"] == 1
    assert math.isnan(summary.loc[("prior_val", 0), "missing_values"])
    assert summary.loc[("prior_val", 0), "duplicates"] == 1
    assert summary.loc[("prior_val", 0), "abs_z_over_100"] == 0
