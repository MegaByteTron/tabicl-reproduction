import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .table import TableSample

TENSORS = ("X_train", "y_train", "X_test", "y_test")


@dataclass(frozen=True, slots=True)
class SavedTablesDataset(IterableDataset[TableSample]):
    """Tables saved as prior_0000.pt, prior_0001.pt, ... (see prior_val.py).

    Read in a fixed order; dataset_id is the table number.
    """

    data_dir: Path

    def __iter__(self) -> Iterator[TableSample]:
        # meta.json is written last, so without it the set is incomplete.
        meta_path = Path(self.data_dir) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No complete set of saved tables in {self.data_dir}; delete "
                "it and run: python -m tabicl_reproduction.prior_val"
            )
        paths = sorted(Path(self.data_dir).glob("prior_*.pt"))
        n_tables = json.loads(meta_path.read_text())["n_tables"]
        if len(paths) != n_tables:
            raise ValueError(
                f"{self.data_dir} has {len(paths)} tables, meta.json says {n_tables}"
            )
        # With DataLoader workers, give each worker its own share of the files.
        worker = get_worker_info()
        if worker is not None:
            paths = paths[worker.id :: worker.num_workers]
        for path in paths:
            tensors = torch.load(path, weights_only=True)
            yield TableSample(
                *(tensors[name] for name in TENSORS),
                dataset_id=int(path.stem.removeprefix("prior_")),
            )
