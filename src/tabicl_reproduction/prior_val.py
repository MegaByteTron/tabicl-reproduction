"""A fixed set of prior tables, saved once, so every model is scored on the same ones.

load_data("prior") makes new random tables on every call, so it can't be used
to compare models. save_prior_tables() writes prior_0000.pt, prior_0001.pt, ...
and meta.json; load_data("prior_val") reads them back.
"""

import json
import random
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabicl_reproduction.data import load_data
from tabicl_reproduction.data.saved_tables import TENSORS
from tabicl_reproduction.results import _git_commit

VERSION = "v1"
N_TABLES = 1600  # the number ticket #7 asks for
SEED = 2026  # pretraining uses seed 0, so these are not training tables
# The pretraining prior settings (PriorData in training/configs.py).
PRIOR_SETTINGS: dict[str, Any] = {
    "n_train": 0.8,
    "n_samples": 1024,
    "min_features": 2,
    "max_features": 100,
    "max_classes": 10,
}


def save_prior_tables(
    out_dir: str | Path, n_tables: int = N_TABLES, seed: int = SEED
) -> None:
    """Generate n_tables prior tables and save them to out_dir."""
    out_dir = Path(out_dir)
    if any(out_dir.glob("prior_*.pt")):
        raise FileExistsError(f"{out_dir} already has prior tables; delete them first")
    out_dir.mkdir(parents=True, exist_ok=True)

    # The prior draws from the global random generators.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tables = load_data("prior", num_workers=0, **PRIOR_SETTINGS)
    for i, table in zip(range(n_tables), tables, strict=False):
        # clone(): a slice would also save the memory of the rest of the table.
        tensors = {name: t.clone() for name, t in zip(TENSORS, table, strict=True)}
        torch.save(tensors, out_dir / f"prior_{i:04d}.pt")

    meta = {
        "version": VERSION,
        "n_tables": n_tables,
        "seed": seed,
        "prior_settings": PRIOR_SETTINGS,
        "git_commit": _git_commit(),
        "versions": {p: version(p) for p in ("torch", "numpy", "scikit-learn")},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    # uv run python -m tabicl_reproduction.prior_val   (once, from the repo root)
    import argparse

    from tabicl_reproduction.data.loader import default_data_dir

    parser = argparse.ArgumentParser(description="Save the fixed prior tables.")
    parser.add_argument("--out", type=Path, default=default_data_dir("prior_val"))
    parser.add_argument("--n-tables", type=int, default=N_TABLES)
    args = parser.parse_args()
    save_prior_tables(args.out, args.n_tables)
    print(f"Saved {args.n_tables} tables to {args.out}")
