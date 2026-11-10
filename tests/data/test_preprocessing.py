import pandas as pd
import torch

from tabicl_reproduction.data.preprocessing import (
    encode_categoricals,
    encode_labels,
    impute,
    resolve_split,
    split_indices,
)

# ----- resolve_split -----


def test_resolve_split_default_is_80_20():
    assert resolve_split(150, None, None) == (120, 30)


def test_resolve_split_fractions_and_counts():
    assert resolve_split(200, 0.5, 0.25) == (100, 50)
    assert resolve_split(200, 100, 50) == (100, 50)
    assert resolve_split(200, None, 50) == (150, 50)


def test_resolve_split_caps_training_rows():
    assert resolve_split(10_000, None, None, max_train=2048) == (2048, 2000)
    assert resolve_split(10_000, 5000, 500, max_train=2048) == (2048, 500)


# ----- encoding and imputation -----


def test_encode_categoricals_keeps_numeric_minus_one():
    X = pd.DataFrame({"num": [-1.0, 2.0, 3.0], "cat": ["a", None, "b"]})
    out = encode_categoricals(X)
    assert out[:, 0].tolist() == [-1.0, 2.0, 3.0]
    assert out[0, 1] == 0 and out[1, 1].isnan() and out[2, 1] == 1


def test_encode_labels_uses_train_classes():
    y_train, y_test = encode_labels(torch.tensor([3, 8, 3, 5]), torch.tensor([5, 9]))
    assert y_train.tolist() == [0, 2, 0, 1]
    assert y_test.tolist() == [1, -1]  # 9 never seen in training


def test_impute_uses_train_mean():
    X_train = torch.tensor([[1.0, float("nan")], [3.0, float("nan")]])
    X_test = torch.tensor([[float("nan"), 5.0]])
    X_train, X_test = impute(X_train, X_test)
    assert X_test.tolist() == [[2.0, 5.0]]
    assert X_train[:, 1].tolist() == [0.0, 0.0]  # all-NaN column -> 0


# ----- split_indices -----


def test_split_is_stratified_and_disjoint():
    y = torch.tensor([0] * 10 + [1] * 10)
    train_idx, test_idx = split_indices(y, n_train=10, n_test=10, seed=0)
    assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())
    assert y[train_idx].bincount().tolist() == [5, 5]


def test_split_falls_back_for_singleton_class():
    y = torch.tensor([0, 0, 0, 1])
    train_idx, test_idx = split_indices(y, n_train=2, n_test=2, seed=0)
    assert len(train_idx) == 2 and len(test_idx) == 2
