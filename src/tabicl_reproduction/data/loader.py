import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, IterableDataset

from .benchmarks import BENCHMARKS
from .openml_tables import OpenMLDataset
from .preprocessing import Size
from .synthetic import PriorDataset
from .table import TableSample

CLUSTER_DATASETS_DIR = Path("/home/space/datasets")
LOCAL_DATASETS_DIR = Path(".cache/datasets")


def default_data_dir(benchmark: str) -> Path:
    """Where a benchmark is stored: <root>/<benchmark>.

    The root is $TABICL_DATASETS_DIR if set, otherwise /home/space/datasets
    when it exists (on the cluster), otherwise .cache/datasets.
    """
    root = os.environ.get("TABICL_DATASETS_DIR")
    if root is None:
        cluster = CLUSTER_DATASETS_DIR.is_dir()
        root = CLUSTER_DATASETS_DIR if cluster else LOCAL_DATASETS_DIR
    return Path(root) / benchmark


type Transformation = Callable[[TableSample], TableSample]


@dataclass(frozen=True, slots=True)
class TransformedDataset(IterableDataset[TableSample]):
    """Applies a transformation lazily to every table of a source."""

    source: IterableDataset[TableSample]
    transformation: Transformation

    def __iter__(self) -> Iterator[TableSample]:
        for sample in self.source:
            # Accept plain tuples from the transformation too, and keep the id.
            yield TableSample(
                *self.transformation(sample), dataset_id=sample.dataset_id
            )


def load_data(
    dataset: str = "prior",
    transformation: Transformation | None = None,
    n_train: Size = None,
    n_test: Size = None,
    **kwargs: Any,
) -> DataLoader[TableSample]:
    """Lazily yield whole tables as (X_train, y_train, X_test, y_test).

    Args:
        dataset: "prior" samples tables from the prior (endless stream).
            Any name in BENCHMARKS (e.g. "tabpfn_val", the TabPFNv2
            classification development datasets) reads those OpenML
            datasets from disk. Each sample has sample.dataset_id.
        transformation: applied to every table, takes and returns a sample.
        n_train: training size per table. int = rows, float = fraction.
            For benchmarks it is capped at 2048 rows.
        n_test: test size per table. int = rows, float = fraction.
            If both are None: 80/20 split.
        **kwargs: passed to PriorDataset or OpenMLDataset
            (e.g. n_samples, max_features, data_dir, seed, download).
            data_dir defaults to default_data_dir(dataset).
    """
    name = dataset.replace(" ", "_")
    if name == "prior":
        source = PriorDataset(n_train=n_train, n_test=n_test, **kwargs)
    elif name in BENCHMARKS:
        kwargs.setdefault("data_dir", default_data_dir(name))
        source = OpenMLDataset(
            dataset_ids=tuple(BENCHMARKS[name].values()),
            n_train=n_train,
            n_test=n_test,
            **kwargs,
        )
    else:
        options = ", ".join(repr(n) for n in ["prior", *BENCHMARKS])
        raise ValueError(f"Unknown dataset {dataset!r}, expected one of {options}")

    if transformation is not None:
        source = TransformedDataset(source, transformation)

    # Each item is already one whole table, and tables differ in shape, so
    # turn off automatic batching (the default collate can't stack them).
    return DataLoader(source, batch_size=None)
