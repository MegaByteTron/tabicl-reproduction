from collections.abc import Iterator
from dataclasses import dataclass

import torch


# eq=False: the generated __eq__ would compare tensors, which raises an error.
@dataclass(frozen=True, slots=True, eq=False)
class TableSample:
    """One classification task: labelled training rows and test rows to predict.

    Unpacks into four values: X_train, y_train, X_test, y_test = sample.
    The other fields are not part of the unpacking:
    - dataset_id: where the table came from (OpenML id, saved prior table
      number, or None for a fresh prior table).
    - split: which train/test split of that dataset (the fold number).
    - class_names: the original label of each class, so label k is
      class_names[k]; None when there are no names (prior tables).

    Labels are consecutive integers 0..n_classes-1 based on the training rows.
    A test row whose class never appears in training has label -1.
    """

    X_train: torch.Tensor  # [n_train, n_features], float32
    y_train: torch.Tensor  # [n_train], int64
    X_test: torch.Tensor  # [n_test, n_features], float32
    y_test: torch.Tensor  # [n_test], int64
    dataset_id: int | None = None
    split: int = 0
    class_names: tuple[str, ...] | None = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter((self.X_train, self.y_train, self.X_test, self.y_test))
