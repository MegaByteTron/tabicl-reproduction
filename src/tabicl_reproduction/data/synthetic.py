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
    structure and number of classes, and its columns in random order.

    Split sizes:
    - n_train and n_test both ints: the table has exactly n_train + n_test
      rows (n_samples and max_samples are ignored).
    - otherwise: the table has n_samples rows, or, if max_samples is set, a
      random number between n_samples and max_samples (log-uniform), split
      by resolve_split (80/20 when n_train and n_test are None).
    """

    n_train: Size = None
    n_test: Size = None
    n_samples: int = 1024
    max_samples: int | None = None
    min_features: int = 2
    max_features: int = 100
    max_classes: int = 10
    max_cat_size: int = 100

    def _split_sizes(self) -> tuple[int, int]:
        if isinstance(self.n_train, int) and isinstance(self.n_test, int):
            return self.n_train, self.n_test
        n_rows = self.n_samples
        if self.max_samples is not None:
            n_rows = randlogint(self.n_samples, self.max_samples + 1)
        return resolve_split(n_rows, self.n_train, self.n_test)

    def sample(self) -> TableSample:
        n_train, n_test = self._split_sizes()
        n_features = randlogint(self.min_features, self.max_features + 1)
        x_cat_sizes = rand_cat_sizes(n_features, self.max_cat_size)
        n_classes = int(np.random.randint(2, self.max_classes + 1))

        columns = rand_dataset_filtered(x_cat_sizes, [n_classes], n_train + n_test)

        # Every column has shape (n_rows, 1). The prior puts numerical columns
        # before categorical ones, and the model depends on column order, so
        # shuffle them, as TabICLv2 does (paper App. E.13).
        order = torch.randperm(n_features).tolist()
        X = torch.cat([columns[f"x_{i}"].float() for i in order], dim=-1)
        y = columns["y_0"].long().squeeze(-1)

        # Rows are drawn i.i.d., so a plain slice is a random split.
        return finalize(X[:n_train], y[:n_train], X[n_train:], y[n_train:])

    def __iter__(self) -> Iterator[TableSample]:
        while True:
            yield self.sample()
