import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import openml
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .benchmarks import TARGET_OVERRIDES
from .preprocessing import (
    Size,
    encode_categoricals,
    finalize,
    resolve_split,
    split_indices,
)
from .table import TableSample


@dataclass(frozen=True, slots=True)
class OpenMLDataset(IterableDataset[TableSample]):
    """Real classification tables from OpenML, cached locally as parquet."""

    dataset_ids: tuple[int, ...]
    data_dir: Path
    n_train: Size = None  # int = rows, float = fraction; see resolve_split
    n_test: Size = None
    max_train: int | None = 2048
    download: bool = True
    seed: int = 42
    # Which random train/test split: 0 is the default split, 1 a second,
    # independent one (TabICLv2 evaluates on two splits per dataset).
    split_seed: int = 0

    def _paths(self, dataset_id: int) -> tuple[Path, Path]:
        directory = self.data_dir / str(dataset_id)
        return (
            directory / "features.parquet",
            directory / "target.parquet",
        )

    def _ensure_cached(self, dataset_id: int) -> tuple[Path, Path]:
        """Download the dataset if it isn't on disk yet; return its file paths."""
        features_path, target_path = self._paths(dataset_id)
        if features_path.exists() and target_path.exists():
            return features_path, target_path

        if not self.download:
            raise FileNotFoundError(
                f"Dataset {dataset_id} is not cached in {self.data_dir}"
            )

        dataset = openml.datasets.get_dataset(dataset_id)
        default_target = dataset.default_target_attribute or ""
        target = TARGET_OVERRIDES.get(dataset_id, default_target)
        if not target or "," in target:
            raise ValueError(
                f"Dataset {dataset_id} ({dataset.name}) has no single target "
                f"column (OpenML default target: {target!r})"
            )
        X, y, _, _ = dataset.get_data(target=target, dataset_format="dataframe")

        assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)
        # The other label columns would give the answer away, so drop them.
        other_labels = [c for c in default_target.split(",") if c != target]
        X = X.drop(columns=other_labels, errors="ignore")

        features_path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(X, features_path)
        _write_parquet(y.to_frame(name="target"), target_path)

        # The data folder is shared on the cluster, so let everyone read it.
        _make_readable(self.data_dir)
        return features_path, target_path

    def _load_dataset(self, dataset_id: int) -> tuple[pd.DataFrame, pd.Series]:
        features_path, target_path = self._ensure_cached(dataset_id)
        return pd.read_parquet(features_path), pd.read_parquet(target_path)["target"]

    def download_all(self) -> dict[int, Exception]:
        """Download every dataset that isn't on disk yet.

        Keeps going when one fails and returns the failures by dataset id.
        """
        failures = {}
        for dataset_id in self.dataset_ids:
            try:
                self._ensure_cached(dataset_id)
            except Exception as error:  # report all failures at the end
                failures[dataset_id] = error
        return failures

    def load(self, dataset_id: int) -> TableSample:
        X_df, y_series = self._load_dataset(dataset_id)

        # Rows without a label can't be trained on or scored, so drop them.
        labeled = y_series.notna().to_numpy()
        X = encode_categoricals(X_df[labeled])
        codes, _ = pd.factorize(y_series[labeled])
        y = torch.tensor(codes, dtype=torch.long)

        n_rows = len(X)
        if n_rows < 2:
            raise ValueError(f"Dataset {dataset_id} has fewer than two rows")

        # Rows are picked at random from the whole table, so large tables
        # are subsampled rather than truncated.
        n_train, n_test = resolve_split(
            n_rows, self.n_train, self.n_test, self.max_train
        )
        # 100_000 is above every OpenML id we use, so no two (dataset, split)
        # pairs share a seed; split_seed=0 keeps the original splits.
        seed = self.seed + dataset_id + 100_000 * self.split_seed
        train_idx, test_idx = split_indices(y, n_train, n_test, seed=seed)
        return finalize(
            X[train_idx],
            y[train_idx],
            X[test_idx],
            y[test_idx],
            dataset_id=dataset_id,
        )

    def __iter__(self) -> Iterator[TableSample]:
        # With DataLoader workers, every worker runs this loop; give each one
        # its own share of the datasets so none is loaded twice.
        worker = get_worker_info()
        dataset_ids = self.dataset_ids
        if worker is not None:
            dataset_ids = dataset_ids[worker.id :: worker.num_workers]
        for dataset_id in dataset_ids:
            yield self.load(dataset_id)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    # Write to a temporary file and rename, so a job that gets killed
    # mid-write never leaves a broken file that later runs would read.
    tmp_path = path.with_suffix(".tmp")
    df.to_parquet(tmp_path)
    tmp_path.replace(path)


# Permission bits for owner, group and others ("a" in chmod).
READ_BY_ALL = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
EXECUTE_BY_ALL = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH


def _make_readable(root: Path) -> None:
    """Like `chmod -R a+rX root`, for the files and folders we own.

    Folders also get the execute bit (the X), which others need to open
    them; with only a+r they could list a folder but not read inside it.
    Paths owned by other users are skipped, since we can't change them.
    """
    uid = os.getuid()
    for path in [root, *root.rglob("*")]:
        info = path.stat()
        if info.st_uid != uid:
            continue
        added = READ_BY_ALL | EXECUTE_BY_ALL if path.is_dir() else READ_BY_ALL
        path.chmod(info.st_mode | added)
