"""Compare every registered model on the same tables.

From the repo root (prior tables first: python -m tabicl_reproduction.data.prior_val):

    uv run python scripts/benchmark.py --limit 5                 # quick test
    uv run python scripts/benchmark.py                           # all baselines
    uv run python scripts/benchmark.py --checkpoint <run dir>    # + our model
    uv run python scripts/benchmark.py --checkpoint <run dir> --models nano_tabicl_v2
    uv run python scripts/benchmark.py --summary                 # summary only

Tables: every tabpfn_val dataset split by 5-fold stratified cross-validation
(at most 2,048 training rows per fold), and the saved prior_val tables (all
1,600 by default) with their fixed split. For each table, every model is
fitted on the training rows, predicts the test rows once, and every
registered metric is computed from those probabilities.

Writes results/<model>.csv, the predictions to results/predictions/<model>/,
and the summary to results/summary.txt (see results.py).

score_table is also what pretraining validation uses (training/evaluator.py),
so validation and the benchmark can only differ by their settings.
"""

import argparse
import contextlib
import itertools
import json
import time
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from tabicl_reproduction.data import BENCHMARKS, OpenMLDataset, TableSample, load_data
from tabicl_reproduction.data.loader import default_data_dir
from tabicl_reproduction.data.prior_val import N_TABLES
from tabicl_reproduction.model.classifier import TabICLClassifier, UnsupportedTableError

from .metrics import evaluate_metrics, to_numpy
from .models import get_models
from .results import ResultsWriter, write_summary

N_FOLDS = 5

# Called with (row, proba, table) after each table; proba is None if skipped.
type OnRow = Callable[[dict[str, Any], np.ndarray | None, TableSample], None]

# ----- scoring one table -----


def table_info(table: TableSample) -> dict[str, Any]:
    """What a result row says about the table itself."""
    return {
        "dataset_id": table.dataset_id,
        "split": table.split,
        "n_train": len(table.y_train),
        "n_test": int((table.y_test >= 0).sum()),  # rows that can be scored
        "n_features": table.X_train.shape[1],
        "n_classes": int(table.y_train.max()) + 1,
    }


def score_table(
    estimator: Any, table: TableSample, skip_errors: bool = False
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Fit on the training rows, predict the test rows once, compute the metrics.

    Returns (row, proba). row has the table info, every registered metric,
    fit_s, predict_s, time_s and status "ok"; or status "skipped" with the
    reason in error, and proba None. A table is skipped when it can't be
    scored fairly (fewer than 2 classes, no scorable test row), when the
    estimator raises UnsupportedTableError, and, with skip_errors, when fit or
    predict raises anything else (baselines on odd tables). A wrong proba
    shape always raises: that is a bug, not a property of the table.
    """
    row = table_info(table)
    # Checked for every model: with one class, most baselines would report
    # a meaningless "perfect" score.
    if row["n_classes"] < 2:
        return row | _skipped("fewer than 2 classes in training"), None
    if row["n_test"] == 0:
        return row | _skipped("no test row with a class seen in training"), None

    X_train, y_train, X_test, _ = (t.numpy() for t in table)
    # Only fit and predict are guarded, so an error in our own code below is
    # never hidden as a "skipped" table.
    try:
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(X_train, y_train)
            fitted = time.perf_counter()
            proba = to_numpy(estimator.predict_proba(X_test))
        done = time.perf_counter()
    except UnsupportedTableError as error:
        return row | _skipped(str(error)), None
    except Exception as error:
        if not skip_errors:
            raise
        return row | _skipped(f"{type(error).__name__}: {error}"), None

    expected = (len(table.y_test), row["n_classes"])
    if proba.shape != expected:
        raise ValueError(
            f"proba must have shape {expected} (test rows, training classes), "
            f"got {proba.shape}"
        )
    classes = getattr(estimator, "classes_", None)
    if classes is not None and not np.array_equal(classes, np.arange(expected[1])):
        raise ValueError(f"proba columns must be classes 0..n-1, got {classes}")

    row |= evaluate_metrics(table.y_test, proba)
    row |= {
        "fit_s": fitted - start,
        "predict_s": done - fitted,
        "time_s": done - start,
        "status": "ok",
        "error": None,
    }
    return row, proba


def score_tables(
    estimator: Any,
    tables: Iterable[TableSample],
    on_row: OnRow | None = None,
    skip_errors: bool = False,
) -> list[dict[str, Any]]:
    """score_table on every table; returns the rows, skipped ones included."""
    rows = []
    for table in tables:
        row, proba = score_table(estimator, table, skip_errors)
        rows.append(row)
        if on_row is not None:
            on_row(row, proba, table)
    return rows


def _skipped(reason: str) -> dict[str, Any]:
    return {"status": "skipped", "error": reason}


# ----- the tables -----


@dataclass(frozen=True)
class Source:
    """A set of tables every model is scored on."""

    name: str
    tables: Iterable[TableSample]
    n_tables: int  # for the progress bar
    data_version: str | None = None


def sources(
    cv: Any, limit: int | None = None, prior_tables: int | None = None
) -> list[Source]:
    """tabpfn_val split by cv, then the saved prior_val tables.

    limit: datasets (and prior tables, unless prior_tables is given) per
    source, for a quick test.
    prior_tables: how many saved prior tables to use (default: all).
    """
    prior_dir = default_data_dir() / "prior_val"
    if not (prior_dir / "meta.json").exists():
        raise SystemExit(
            f"No complete prior tables in {prior_dir}; delete the folder if it "
            "exists and run: python -m tabicl_reproduction.data.prior_val"
        )
    meta = json.loads((prior_dir / "meta.json").read_text())
    if limit is None and prior_tables is None and meta["n_tables"] != N_TABLES:
        # e.g. a small test set left over from trying the command out
        raise SystemExit(
            f"{prior_dir} has {meta['n_tables']} tables, the full run needs "
            f"{N_TABLES}; delete it and run: python -m tabicl_reproduction.data.prior_val"
        )

    # Download every OpenML dataset first, so a failure shows up right away
    # instead of in the middle of the run.
    datasets = dict(itertools.islice(BENCHMARKS["tabpfn_val"].items(), limit))
    if failures := OpenMLDataset(datasets, default_data_dir()).download_all():
        raise SystemExit(f"Could not download: {failures}")

    n_openml = len(datasets) * cv.get_n_splits()
    # An explicit prior_tables wins over limit.
    n_prior = min(meta["n_tables"], prior_tables or limit or meta["n_tables"])
    return [
        Source(
            "tabpfn_val", _First(load_data("tabpfn_val", cv=cv), n_openml), n_openml
        ),
        Source(
            "prior_val",
            _First(load_data("prior_val"), n_prior),
            n_prior,
            data_version=meta["version"],
        ),
    ]


class _First:
    """The first n tables of a re-iterable source, re-iterable too."""

    def __init__(self, tables: Iterable[TableSample], n: int):
        self.tables, self.n = tables, n

    def __iter__(self) -> Iterator[TableSample]:
        return itertools.islice(iter(self.tables), self.n)


# ----- the benchmark -----


def run_benchmark(
    models: dict[str, Any],
    table_sources: list[Source],
    out: Path,
    save_predictions: bool = True,
) -> None:
    """Score every model on every table; one CSV per model in out/."""
    taken = [name for name in models if (out / f"{name}.csv").exists()]
    if taken:
        # The writers append, so a second run would add duplicate rows.
        raise SystemExit(f"{out}/ already has results for {taken}; move them first")
    writers = {
        name: ResultsWriter(
            out / f"{name}.csv",
            model=name,
            checkpoint_step=getattr(model, "checkpoint_step", None),
            predictions_dir=out / "predictions" / name if save_predictions else None,
        )
        for name, model in models.items()
    }

    for source in table_sources:
        tables = tqdm(source.tables, total=source.n_tables, desc=source.name)
        for table in tables:  # each table is loaded once, for every model
            for name, model in models.items():
                ours = isinstance(model, TabICLClassifier)
                # On these small tables, extra threads only slow the baselines
                # down (one thread is ~7x faster); our model uses torch's.
                threads = contextlib.nullcontext() if ours else threadpool_limits(1)
                with threads:
                    # A baseline failing on an odd table is a result; our
                    # model failing is a bug, so it raises.
                    row, proba = score_table(model, table, skip_errors=not ours)
                writers[name].add(
                    row,
                    proba,
                    table,
                    source=source.name,
                    data_version=source.data_version,
                )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare models on the same tables.")
    parser.add_argument(
        "--checkpoint", type=Path, help="training run folder (our model)"
    )
    parser.add_argument("--weights", choices=["best", "model"], default="best")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-orders", type=int, default=8, help="column orders (ours)")
    parser.add_argument("--models", nargs="+", help="default: every registered model")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--prior-tables", type=int, help=f"default: all {N_TABLES}")
    parser.add_argument("--limit", type=int, help="datasets per source, quick test")
    parser.add_argument(
        "--out", type=Path, help="default: results/ (results_test/ with --limit)"
    )
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--summary", action="store_true", help="only print the summary")
    args = parser.parse_args(argv)
    out = args.out or Path("results_test" if args.limit else "results")

    if not args.summary:
        try:
            models = dict(
                get_models(
                    args.models,
                    checkpoint=args.checkpoint,
                    weights=args.weights,
                    device=args.device,
                    n_orders=args.n_orders,
                )
            )
        except ValueError as error:
            raise SystemExit(error) from None
        cv = StratifiedKFold(args.folds, shuffle=True, random_state=0)
        table_sources = sources(cv, args.limit, args.prior_tables)
        run_benchmark(models, table_sources, out, not args.no_predictions)
    try:
        write_summary(out)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(error) from None


if __name__ == "__main__":
    main()
