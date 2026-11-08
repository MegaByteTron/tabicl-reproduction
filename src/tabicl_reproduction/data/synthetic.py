from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset

from tabicl_reproduction.prior import rand_cat_sizes, rand_dataset_filtered, randlogint

from .preprocessing import Size, finalize, resolve_split
from .table import TableSample


@dataclass(frozen=True, slots=True)
class PriorDataset(IterableDataset[TableSample]):
    """Endless stream of synthetic classification tables from the prior.

    Each table gets a newly sampled number of features, categorical
    structure and number of classes.

    Split sizes:
    - n_train and n_test both ints: the table has exactly n_train + n_test
      rows (n_samples is ignored).
    - otherwise: the table has n_samples rows, split by resolve_split
      (80/20 when both are None).
    """

    n_train: Size = None
    n_test: Size = None
    n_samples: int = 1024
    min_features: int = 2
    max_features: int = 100
    max_classes: int = 10
    max_cat_size: int = 100

    def _split_sizes(self) -> tuple[int, int]:
        if isinstance(self.n_train, int) and isinstance(self.n_test, int):
            return self.n_train, self.n_test
        return resolve_split(self.n_samples, self.n_train, self.n_test)

    def sample(self) -> TableSample:
        n_train, n_test = self._split_sizes()
        n_features = randlogint(self.min_features, self.max_features + 1)
        x_cat_sizes = rand_cat_sizes(n_features, self.max_cat_size)
        n_classes = int(np.random.randint(2, self.max_classes + 1))

        columns = rand_dataset_filtered(x_cat_sizes, [n_classes], n_train + n_test)

        # Every column has shape (n_rows, 1); keep the x_i order.
        X = torch.cat([columns[f"x_{i}"].float() for i in range(n_features)], dim=-1)
        y = columns["y_0"].long().squeeze(-1)

        # Rows are drawn i.i.d., so a plain slice is a random split.
        return finalize(X[:n_train], y[:n_train], X[n_train:], y[n_train:])

    def __iter__(self) -> Iterator[TableSample]:
        while True:
            yield self.sample()
