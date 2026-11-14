"""Score an estimator on tables: the only place where a table gets scored.

    rows = score_tables(estimator, tables)          # just the scores
    save = partial(writer.add, source="tabpfn_val")
    score_tables(estimator, tables, on_row=save)    # also save them

Pretraining validation (training/evaluator.py) and the experiment
(evaluation/experiment.py) both call score_tables, so their numbers can only
differ by the settings passed in: which estimator, which tables. Nothing here
writes files; on_row decides what happens with each result.

The estimator is anything with fit(X, y) and predict_proba(X): our
TabICLClassifier or a scikit-learn model. It is fitted again on every table
(fit resets it), not cloned: cloning would copy our whole network each time.
"""

import time
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from tabicl_reproduction.data import TableSample
from tabicl_reproduction.model.classifier import UnsupportedTableError

from .metrics import compute_metrics, to_numpy

# Called with (row, proba, table) after each table; proba is None if skipped.
type OnRow = Callable[[dict[str, Any], np.ndarray | None, TableSample], None]


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
    """Fit on the training rows, predict the test rows, compute the metrics.

    Returns (row, proba). row has the table info, the metrics, fit_s,
    predict_s, time_s and status "ok"; or status "skipped" with the reason
    in error, and proba None. A table is skipped when it can't be scored
    fairly (fewer than 2 classes, no scorable test row), when the estimator
    raises UnsupportedTableError, and, with skip_errors, when fit or predict
    raises anything else (for baselines on odd tables). A wrong proba shape
    always raises: that is a bug, not a property of the table.
    """
    row = table_info(table)
    # Checked for every estimator: with one class, most baselines would
    # report a meaningless "perfect" score.
    if row["n_classes"] < 2:
        return row | _skipped("fewer than 2 classes in training"), None
    if row["n_test"] == 0:
        return row | _skipped("no test row with a class seen in training"), None

    X_train, y_train, X_test = (
        t.numpy() for t in (table.X_train, table.y_train, table.X_test)
    )
    # Only fit and predict are guarded, so an error in our own code below
    # is never hidden as a "skipped" table.
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

    row |= compute_metrics(table.y_test, proba)
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
