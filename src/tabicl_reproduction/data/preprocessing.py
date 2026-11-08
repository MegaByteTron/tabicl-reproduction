"""Preprocessing shared by all table sources.

Everything that depends on the data (label mapping, imputation values) is
computed from the training rows only, so no test information leaks in.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .table import TableSample

type Size = int | float | None

DEFAULT_TEST_FRACTION = 0.2


def resolve_split(
    n_rows: int, n_train: Size, n_test: Size, max_train: int | None = None
) -> tuple[int, int]:
    """Turn requested train/test sizes into row counts.

    A float is a fraction of the table, an int is a number of rows.
    - n_test=None: all rows not used for training (or 20% if n_train is
      also None).
    - n_train=None: all rows not used for testing.
    The training set is then capped at max_train rows.
    """

    def to_rows(size: int | float) -> int:
        return round(size * n_rows) if isinstance(size, float) else size

    if n_test is not None:
        test_rows = to_rows(n_test)
    elif n_train is not None:
        test_rows = n_rows - to_rows(n_train)
    else:
        test_rows = round(DEFAULT_TEST_FRACTION * n_rows)
    test_rows = min(max(test_rows, 1), n_rows - 1)

    train_rows = n_rows - test_rows if n_train is None else to_rows(n_train)
    train_rows = min(train_rows, n_rows - test_rows)
    if max_train is not None:
        train_rows = min(train_rows, max_train)

    if train_rows < 1:
        raise ValueError(
            f"No training rows left: n_rows={n_rows}, "
            f"n_train={n_train}, n_test={n_test}"
        )
    return train_rows, test_rows


def encode_categoricals(X: pd.DataFrame) -> torch.Tensor:
    """Turn a DataFrame into a float tensor; categories become integer codes.

    Missing values (in any column) become NaN.
    """
    X = X.copy()
    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            codes = X[column].astype("category").cat.codes
            # cat.codes marks missing values as -1. Turn only those into NaN
            # so real -1 values in numeric columns stay intact.
            X[column] = codes.where(codes >= 0, np.nan)
    return torch.tensor(X.to_numpy(dtype=np.float32, na_value=np.nan))


def split_indices(
    y: torch.Tensor, n_train: int, n_test: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick random train and test rows, stratified by label when possible.

    Stratifying keeps every class present in the training rows, which matters
    for small datasets. It needs at least two rows per class, so rare-class
    datasets fall back to a plain random split.
    """
    indices = np.arange(len(y))
    try:
        train_idx, test_idx = train_test_split(
            indices,
            train_size=n_train,
            test_size=n_test,
            stratify=y.numpy(),
            random_state=seed,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices, train_size=n_train, test_size=n_test, random_state=seed
        )
    return torch.from_numpy(train_idx), torch.from_numpy(test_idx)


def encode_split_categoricals(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit category vocabularies on training rows; unknown test values become NaN."""
    train, test = train.copy(), test.copy()
    for column in train.columns:
        if not pd.api.types.is_numeric_dtype(train[column]):
            # Ignore unused categorical levels inherited from the full table.
            vocabulary = pd.unique(train[column].dropna().astype(object))
            for frame in (train, test):
                codes = pd.Categorical(frame[column], categories=vocabulary).codes
                frame[column] = np.where(codes >= 0, codes, np.nan)
    return (
        torch.tensor(train.to_numpy(dtype=np.float32, na_value=np.nan)),
        torch.tensor(test.to_numpy(dtype=np.float32, na_value=np.nan)),
    )


def encode_labels(
    y_train: torch.Tensor, y_test: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map labels to 0..n_classes-1 using the classes seen in training.

    Test labels of classes missing from training become -1. Use
    ignore_index=-1 in the loss; for accuracy they always count as wrong.
    """
    classes = torch.unique(y_train)  # sorted
    train_enc = torch.searchsorted(classes, y_train)

    test_idx = torch.searchsorted(classes, y_test).clamp(max=len(classes) - 1)
    seen = classes[test_idx] == y_test
    test_enc = torch.where(seen, test_idx, -1)
    return train_enc, test_enc


def impute(
    X_train: torch.Tensor, X_test: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace NaN with the training-row mean of each column.

    Columns that are entirely NaN in training get 0.
    """
    mean = torch.nanmean(X_train, dim=0, keepdim=True).nan_to_num(0.0)
    return (
        torch.where(X_train.isnan(), mean, X_train),
        torch.where(X_test.isnan(), mean, X_test),
    )


def finalize(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    dataset_id: int | None = None,
) -> TableSample:
    """Apply label encoding and imputation to an already split table."""
    y_train, y_test = encode_labels(y_train, y_test)
    X_train, X_test = impute(X_train, X_test)
    return TableSample(X_train, y_train, X_test, y_test, dataset_id=dataset_id)
