"""Simple baselines, fitted from scratch on every table.

Our model must beat these on prior tables (Milestone 1). Every baseline uses
scikit-learn defaults without tuning; k-NN and logistic regression get
standardized features, because they depend on feature scale.

From the repo root, after `uv run python -m tabicl_reproduction.prior_val`:

    uv run python -m tabicl_reproduction.baselines --limit 20   # quick test
    uv run python -m tabicl_reproduction.baselines              # full run
    uv run python -m tabicl_reproduction.baselines --summary    # summary only

The full run scores tabpfn_val (2 splits) and the saved prior tables, writes
results/<baseline>.csv, then prints the summary and saves results/summary.txt.
"""

import argparse
import itertools
import json
import logging
import time
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from tabicl_reproduction.data import BENCHMARKS, OpenMLDataset, TableSample, load_data
from tabicl_reproduction.data.loader import default_data_dir
from tabicl_reproduction.prior_val import N_TABLES
from tabicl_reproduction.results import (
    METRICS,
    ResultsWriter,
    load_results,
    make_table_id,
)

log = logging.getLogger(__name__)

# Each entry makes a new, unfitted model (anything with fit and predict_proba).
BASELINES: dict[str, Callable[[], Any]] = {
    "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "logistic_regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(random_state=0)
    ),
    "random_forest": lambda: RandomForestClassifier(random_state=0),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=0),
}


def run_baselines(
    tables: Iterable[TableSample],
    out_dir: str | Path,
    *,
    source: str,
    split_seed: int = 0,
    data_version: str | None = None,
    baselines: dict[str, Callable[[], Any]] = BASELINES,
) -> None:
    """Fit a new model of each baseline on every table and save its scores.

    Writes one CSV per baseline, out_dir/<name>.csv, with one row per table.
    """
    writers = {
        name: ResultsWriter(Path(out_dir) / f"{name}.csv", model=name)
        for name in baselines
    }
    for table in tables:
        table_id = make_table_id(source, table.dataset_id)
        info = {
            "source": source,
            "table_id": table_id,
            "split_seed": split_seed,
            "data_version": data_version,
        }

        # Checked up front: with one class, only logistic regression raises;
        # the others would return a meaningless "perfect" score.
        if table.y_train.max() < 1:
            reason = "fewer than 2 classes in training"
            log.warning("Skipping %s %s: %s", source, table_id, reason)
            for writer in writers.values():
                writer.skip(table, reason, **info)
            continue

        X_train, y_train, X_test = (
            t.numpy() for t in (table.X_train, table.y_train, table.X_test)
        )
        for name, make_model in baselines.items():
            start = time.perf_counter()
            # Only fit and predict are guarded: an error in our own code
            # below must not be hidden as a "skipped" table.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    proba = make_model().fit(X_train, y_train).predict_proba(X_test)
            except Exception as error:
                log.warning("%s failed on %s %s: %r", name, source, table_id, error)
                writers[name].skip(table, f"{type(error).__name__}: {error}", **info)
                continue
            writers[name].add(table, proba, time_s=time.perf_counter() - start, **info)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric per model and source.

    Only tables that every model scored are counted, so all models are
    compared on the same tables. NaN metrics (e.g. ROC AUC with one class in
    the test rows) are left out of the mean. df comes from results.load_results().
    """
    ok = df[df["status"] == "ok"]
    # A model at one checkpoint is one "run"; dropna=False keeps the
    # baselines, which have an empty checkpoint_step.
    run = ok.groupby(["model", "checkpoint_step"], dropna=False).ngroup()
    ok = ok.assign(run=run)
    # Per source, keep only the tables that every run scored.
    key = ["source", "table_id", "split_seed"]
    runs_per_table = ok.groupby(key)["run"].transform("nunique")
    runs_per_source = ok.groupby("source")["run"].transform("nunique")
    common = ok[runs_per_table == runs_per_source]
    if common.duplicated(["run", *key]).any():
        raise ValueError("A model has two rows for the same table; check the CSVs")

    # dropna=False: baselines have an empty checkpoint_step, which pandas
    # would otherwise drop from the groups.
    groups = common.groupby(["model", "checkpoint_step", "source"], dropna=False)
    summary = groups[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_tables", groups.size())
    return summary.reset_index()


def run_all(out: Path, limit: int | None = None) -> None:
    """Run every baseline on tabpfn_val (2 splits) and the saved prior tables."""
    if any(out.glob("*.csv")):
        # The writers append, so a second run would add duplicate rows.
        raise SystemExit(f"{out}/ already has results; move or delete them first")
    prior_dir = default_data_dir() / "prior_val"
    if not (prior_dir / "meta.json").exists():
        raise SystemExit(
            f"No complete prior tables in {prior_dir}; delete the folder if it "
            "exists and run: python -m tabicl_reproduction.prior_val"
        )
    meta = json.loads((prior_dir / "meta.json").read_text())
    if limit is None and meta["n_tables"] != N_TABLES:
        # e.g. a small test set left over from trying the command out
        raise SystemExit(
            f"{prior_dir} has {meta['n_tables']} tables, the full run needs "
            f"{N_TABLES}; delete it and run: python -m tabicl_reproduction.prior_val"
        )

    # Download every OpenML dataset first, so a failure shows up right away
    # instead of after the long prior_val part.
    datasets = dict(itertools.islice(BENCHMARKS["tabpfn_val"].items(), limit))
    failures = OpenMLDataset(datasets, default_data_dir()).download_all()
    if failures:
        raise SystemExit(f"Could not download: {failures}")

    runs = [("tabpfn_val", s, len(datasets), None) for s in (0, 1)]
    runs.append(("prior_val", 0, meta["n_tables"], meta["version"]))
    for source, split_seed, n_tables, data_version in runs:
        # prior_val tables have one fixed split; OpenML tables get split_seed.
        kwargs: dict[str, Any] = (
            {"split_seed": split_seed} if source == "tabpfn_val" else {}
        )
        tables = load_data(source, **kwargs)
        n = min(n_tables, limit or n_tables)
        run_baselines(
            tqdm(itertools.islice(tables, n), total=n, desc=f"{source} {split_seed}"),
            out,
            source=source,
            split_seed=split_seed,
            data_version=data_version,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize the baselines.")
    parser.add_argument("--limit", type=int, help="tables per source, quick test")
    parser.add_argument(
        "--out", type=Path, help="default: results/ (results_test/ with --limit)"
    )
    parser.add_argument("--summary", action="store_true", help="only print summary")
    args = parser.parse_args()
    out = args.out or Path("results_test" if args.limit else "results")

    if not args.summary:
        # On these small tables, extra threads only wait on each other:
        # one thread is ~7x faster here, with identical results.
        with threadpool_limits(1):
            run_all(out, args.limit)
    try:
        summary = summarize(load_results(out / "*.csv"))
    except FileNotFoundError as error:
        raise SystemExit(error) from None
    if summary.empty:
        raise SystemExit("No table was scored by every model")
    text = summary.to_string(index=False, float_format="%.3f")
    # .txt, not .csv: the folder's *.csv files must all be result files.
    (out / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s")
    main()
