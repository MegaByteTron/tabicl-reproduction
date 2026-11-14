from collections.abc import Iterator
from dataclasses import dataclass

import torch


# eq=False: the generated __eq__ would compare tensors, which raises an error.
@dataclass(frozen=True, slots=True, eq=False)
class TableSample:
    """One classification task: labelled training rows and test rows to predict.

    Unpacks into four values: X_train, y_train, X_test, y_test = sample.
    dataset_id says where the table came from (OpenML id, or None for the
    prior) and is not part of the unpacking.

    Labels are consecutive integers 0..n_classes-1 based on the training rows.
    A test row whose class never appears in training has label -1.
    """

    X_train: torch.Tensor  # [n_train, n_features], float32
    y_train: torch.Tensor  # [n_train], int64
    X_test: torch.Tensor  # [n_test, n_features], float32
    y_test: torch.Tensor  # [n_test], int64
    dataset_id: int | None = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter((self.X_train, self.y_train, self.X_test, self.y_test))
