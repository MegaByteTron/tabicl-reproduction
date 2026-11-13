import os
import stat
from collections.abc import Iterator, Mapping
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
    """Real classification tables from OpenML, cached locally as parquet.

    Each dataset is stored by name: <data_dir>/<name>/{features,target}.parquet.
    """

    datasets: Mapping[str, int]  # dataset name -> OpenML id
    data_dir: Path
    n_train: Size = None  # int = rows, float = fraction; see resolve_split
    n_test: Size = None
    max_train: int | None = 2048
    download: bool = True
    seed: int = 42

    def _paths(self, name: str) -> tuple[Path, Path]:
        directory = self.data_dir / name
        return (
            directory / "features.parquet",
            directory / "target.parquet",
        )

    def _ensure_cached(self, name: str) -> tuple[Path, Path]:
        """Download the dataset if it isn't on disk yet; return its file paths."""
        features_path, target_path = self._paths(name)
        if features_path.exists() and target_path.exists():
            return features_path, target_path

        if not self.download:
            raise FileNotFoundError(f"Dataset {name} is not cached in {self.data_dir}")

        dataset_id = self.datasets[name]

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
        # Only touch this dataset's folder: the root holds other people's data.
        _make_readable(self.data_dir, recursive=False)
        _make_readable(features_path.parent)
        return features_path, target_path

    def _load_dataset(self, name: str) -> tuple[pd.DataFrame, pd.Series]:
        features_path, target_path = self._ensure_cached(name)
        return pd.read_parquet(features_path), pd.read_parquet(target_path)["target"]

    def download_all(self) -> dict[str, Exception]:
        """Download every dataset that isn't on disk yet.

        Keeps going when one fails and returns the failures by dataset name.
        """
        failures = {}
        for name in self.datasets:
            try:
                self._ensure_cached(name)
            except Exception as error:  # report all failures at the end
                failures[name] = error
        return failures

    def load(self, name: str) -> TableSample:
        dataset_id = self.datasets[name]
        X_df, y_series = self._load_dataset(name)

        # Rows without a label can't be trained on or scored, so drop them.
        labeled = y_series.notna().to_numpy()
        X = encode_categoricals(X_df[labeled])
        codes, _ = pd.factorize(y_series[labeled])
        y = torch.tensor(codes, dtype=torch.long)

        n_rows = len(X)
        if n_rows < 2:
            raise ValueError(f"Dataset {name} has fewer than two rows")

        # Rows are picked at random from the whole table, so large tables
        # are subsampled rather than truncated.
        n_train, n_test = resolve_split(
            n_rows, self.n_train, self.n_test, self.max_train
        )
        train_idx, test_idx = split_indices(
            y, n_train, n_test, seed=self.seed + dataset_id
        )
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
        names = list(self.datasets)
        if worker is not None:
            names = names[worker.id :: worker.num_workers]
        for name in names:
            yield self.load(name)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    # Write to a temporary file and rename, so a job that gets killed
    # mid-write never leaves a broken file that later runs would read.
    tmp_path = path.with_suffix(".tmp")
    df.to_parquet(tmp_path)
    tmp_path.replace(path)


# Permission bits for owner, group and others ("a" in chmod).
READ_BY_ALL = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
EXECUTE_BY_ALL = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH


def _make_readable(root: Path, recursive: bool = True) -> None:
    """Like `chmod -R a+rX root`, for the files and folders we own.

    Folders also get the execute bit (the X), which others need to open
    them; with only a+r they could list a folder but not read inside it.
    Paths owned by other users are skipped, since we can't change them.
    """
    uid = os.getuid()
    for path in [root, *root.rglob("*")] if recursive else [root]:
        info = path.stat()
        if info.st_uid != uid:
            continue
        added = READ_BY_ALL | EXECUTE_BY_ALL if path.is_dir() else READ_BY_ALL
        path.chmod(info.st_mode | added)
