"""Compare models on the same tables: our pretrained model and the baselines.

From the repo root (prior tables first: python -m tabicl_reproduction.data.prior_val):

    uv run python scripts/compare.py --limit 5                    # quick test
    uv run python scripts/compare.py                              # baselines
    uv run python scripts/compare.py --checkpoint <run dir>       # + ours (best.pt)
    uv run python scripts/compare.py --checkpoint <run dir> --models tabicl_best
    uv run python scripts/compare.py --summary                    # summary only

Tables: every tabpfn_val dataset split by 5-fold stratified cross-validation
(at most 2,048 training rows per fold), and the saved prior_val tables with
their fixed split. Every model is scored on exactly these, by score_tables.
Writes results/<model>.csv, the predictions to results/predictions/<model>/,
and the summary to results/summary.txt.
"""

import argparse
import contextlib
import itertools
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from tabicl_reproduction.data import BENCHMARKS, OpenMLDataset, TableSample
from tabicl_reproduction.data.loader import default_data_dir
from tabicl_reproduction.data.prior_val import N_TABLES
from tabicl_reproduction.data.saved_tables import SavedTablesDataset
from tabicl_reproduction.model.classifier import TabICLClassifier

from .baselines import make_baselines
from .core import score_tables
from .results import ResultsWriter, write_summary

N_FOLDS = 5


@dataclass(frozen=True)
class Source:
    """A set of tables every model is scored on."""

    name: str
    tables: Iterable[TableSample]  # re-iterable: read once per model
    n_tables: int
    data_version: str | None = None


def sources(cv: Any, limit: int | None = None) -> list[Source]:
    """tabpfn_val split by cv, then the saved prior_val tables.

    limit: datasets (and prior tables) per source, for a quick test.
    """
    prior_dir = default_data_dir() / "prior_val"
    if not (prior_dir / "meta.json").exists():
        raise SystemExit(
            f"No complete prior tables in {prior_dir}; delete the folder if it "
            "exists and run: python -m tabicl_reproduction.data.prior_val"
        )
    meta = json.loads((prior_dir / "meta.json").read_text())
    if limit is None and meta["n_tables"] != N_TABLES:
        # e.g. a small test set left over from trying the command out
        raise SystemExit(
            f"{prior_dir} has {meta['n_tables']} tables, the full run needs "
            f"{N_TABLES}; delete it and run: python -m tabicl_reproduction.data.prior_val"
        )

    # Download every OpenML dataset first, so a failure shows up right away
    # instead of after the long prior_val part.
    datasets = dict(itertools.islice(BENCHMARKS["tabpfn_val"].items(), limit))
    openml = OpenMLDataset(datasets, default_data_dir(), cv=cv)
    if failures := openml.download_all():
        raise SystemExit(f"Could not download: {failures}")

    n_prior = min(meta["n_tables"], limit or meta["n_tables"])
    prior = _Limited(SavedTablesDataset(prior_dir), n_prior)
    return [
        Source("tabpfn_val", openml, len(datasets) * cv.get_n_splits()),
        Source("prior_val", prior, n_prior, data_version=meta["version"]),
    ]


def run(
    estimators: dict[str, Any],
    out: Path,
    table_sources: list[Source],
    save_predictions: bool = True,
) -> None:
    """Score every estimator on every source; one CSV per estimator in out/."""
    taken = [name for name in estimators if (out / f"{name}.csv").exists()]
    if taken:
        # The writers append, so a second run would add duplicate rows.
        raise SystemExit(f"{out}/ already has results for {taken}; move them first")

    for name, estimator in estimators.items():
        writer = ResultsWriter(
            out / f"{name}.csv",
            model=name,
            checkpoint_step=getattr(estimator, "checkpoint_step", None),
            predictions_dir=out / "predictions" / name if save_predictions else None,
        )
        ours = isinstance(estimator, TabICLClassifier)
        # On these small tables, extra threads only slow the baselines down
        # (one thread is ~7x faster, same results); our model uses torch's.
        threads = contextlib.nullcontext() if ours else threadpool_limits(1)
        with threads:
            for source in table_sources:
                tables = tqdm(
                    source.tables, total=source.n_tables, desc=f"{name} {source.name}"
                )
                save = partial(
                    writer.add, source=source.name, data_version=source.data_version
                )
                # A baseline failing on an odd table is a result; our model
                # failing is a bug, so it raises.
                score_tables(estimator, tables, on_row=save, skip_errors=not ours)


class _Limited:
    """The first n tables of a re-iterable source, re-iterable too."""

    def __init__(self, tables: Iterable[TableSample], n: int):
        self.tables, self.n = tables, n

    def __iter__(self):
        return itertools.islice(iter(self.tables), self.n)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare models on the same tables.")
    parser.add_argument("--checkpoint", type=Path, help="training run folder (ours)")
    parser.add_argument("--weights", choices=["best", "model"], default="best")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-orders", type=int, default=8, help="column orders (ours)")
    parser.add_argument("--models", nargs="+", help="default: all available")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--limit", type=int, help="datasets per source, quick test")
    parser.add_argument(
        "--out", type=Path, help="default: results/ (results_test/ with --limit)"
    )
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--summary", action="store_true", help="only print the summary")
    args = parser.parse_args(argv)
    out = args.out or Path("results_test" if args.limit else "results")

    if not args.summary:
        makers: dict[str, Callable[[], Any]] = {
            name: (lambda est=est: est) for name, est in make_baselines().items()
        }
        if args.checkpoint is not None:
            makers[f"tabicl_{args.weights}"] = lambda: TabICLClassifier.from_checkpoint(
                args.checkpoint,
                args.weights,
                device=args.device,
                n_orders=args.n_orders,
            )
        names = args.models or list(makers)
        unknown = sorted(set(names) - set(makers))
        if unknown:
            raise SystemExit(f"Unknown models {unknown}; available: {list(makers)}")

        cv = StratifiedKFold(args.folds, shuffle=True, random_state=0)
        estimators = {name: makers[name]() for name in names}
        run(estimators, out, sources(cv, args.limit), not args.no_predictions)
    try:
        write_summary(out)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(error) from None
