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


def on_cluster() -> bool:
    """True inside a SLURM job, including inside its apptainer container.

    The login node has no python, so all our code on the cluster runs in a job.
    """
    return "SLURM_JOB_ID" in os.environ


def default_data_dir() -> Path:
    """Folder that holds every OpenML dataset, each in <folder>/<name>.

    $TABICL_DATASETS_DIR if set, otherwise /home/space/datasets in a SLURM
    job, otherwise .cache/datasets.
    """
    root = os.environ.get("TABICL_DATASETS_DIR")
    if root is not None:
        return Path(root)
    if not on_cluster():
        return LOCAL_DATASETS_DIR
    # Fail loudly: falling back to .cache would download into the job's
    # per-commit worktree instead of the shared folder.
    if not CLUSTER_DATASETS_DIR.is_dir():
        raise FileNotFoundError(
            f"In a SLURM job but {CLUSTER_DATASETS_DIR} is not visible. Bind it "
            f"into the container (-B {CLUSTER_DATASETS_DIR}) or set "
            "TABICL_DATASETS_DIR."
        )
    return CLUSTER_DATASETS_DIR


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
    num_workers: int = 0,
    prefetch_factor: int = 4,
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
        num_workers: processes producing tables in parallel (0 = in this
            process). The prior is CPU-bound, so this keeps a GPU busy.
            PyTorch gives every worker its own random, numpy and torch seed.
        prefetch_factor: tables each worker prepares ahead.
        **kwargs: passed to PriorDataset or OpenMLDataset
            (e.g. n_samples, max_features, data_dir, seed, download).
            data_dir defaults to default_data_dir().
    """
    name = dataset.replace(" ", "_")
    if name == "prior":
        source = PriorDataset(n_train=n_train, n_test=n_test, **kwargs)
    elif name in BENCHMARKS:
        kwargs.setdefault("data_dir", default_data_dir())
        source = OpenMLDataset(
            datasets=BENCHMARKS[name],
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
    return DataLoader(
        source,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
