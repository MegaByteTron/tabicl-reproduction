"""Saving and reading benchmark results: one CSV per model, plus predictions.

    df = load_results("results/*.csv")           # every model in one DataFrame
    summary = summarize(df)                       # what summary.txt shows
    p = load_predictions("results/predictions/random_forest/tabpfn_val/31_split0.npz")

Columns (one row per model, table and split; empty cells load as NaN):
model, source, table_id, split, data_version (prior tables only),
checkpoint_step (our model only), n_train, n_test, n_features, n_classes,
one column per registered metric, fit_s, predict_s, time_s,
status (ok or skipped), error (why skipped), git_commit.

Predictions: <predictions_dir>/<source>/<table_id>_split<k>.npz with y_true
(labels 0..n_classes-1, -1 for a class unseen in training), proba
([n_test, n_classes], column k for label k) and class_names (the original
label of each class).
"""

import csv
import glob
from collections.abc import Hashable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabicl_reproduction.data import TableSample
from tabicl_reproduction.utils import git_commit

from .metrics import METRIC_REGISTRY, metric_names

COLUMNS = (
    "model",
    "source",
    "table_id",
    "split",
    "data_version",
    "checkpoint_step",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    *metric_names(),
    "fit_s",
    "predict_s",
    "time_s",
    "status",
    "error",
    "git_commit",
)
# Read as text, so ids like "31" or a commit like "1234567" stay strings.
TEXT_COLUMNS = (
    "model",
    "source",
    "table_id",
    "data_version",
    "status",
    "error",
    "git_commit",
)
INT_COLUMNS = ("split", "n_train", "n_test", "n_features", "n_classes")
# Fixed types, so a file without rows can't turn numbers into text when
# files are combined. All other columns are floats (empty cells become NaN).
DTYPES: dict[Hashable, str] = {
    c: "str" if c in TEXT_COLUMNS else "int64" if c in INT_COLUMNS else "float64"
    for c in COLUMNS
}
# Results are matched across models by these columns.
TABLE_KEY = ["source", "table_id", "split"]


def make_table_id(source: str, dataset_id: int | None) -> str:
    """The table_id every model writes: "prior_0017" or an OpenML id "31"."""
    if dataset_id is None:
        raise ValueError("The table has no dataset_id")
    return f"prior_{dataset_id:04d}" if source == "prior_val" else str(dataset_id)


class ResultsWriter:
    """Appends one row per scored table to a CSV, and saves the predictions.

    Each row is written right away, so nothing is lost if a run stops.
    """

    def __init__(
        self,
        path: str | Path,
        model: str,
        checkpoint_step: int | None = None,
        predictions_dir: str | Path | None = None,
    ):
        self.path = Path(path)
        self.model = model
        self.checkpoint_step = checkpoint_step
        self.predictions_dir = (
            None if predictions_dir is None else Path(predictions_dir)
        )
        self.git_commit = git_commit()

        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open(newline="") as f:
                header = next(csv.reader(f), [])
            if tuple(header) != COLUMNS:
                raise ValueError(f"{self.path} has other columns than COLUMNS")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(COLUMNS)

    def add(
        self,
        row: dict[str, Any],
        proba: np.ndarray | None = None,
        table: TableSample | None = None,
        *,
        source: str,
        data_version: str | None = None,
    ) -> None:
        """Save a row from score_table, and its predictions if given."""
        table_id = make_table_id(source, row["dataset_id"])
        full = {c: row[c] for c in COLUMNS if c in row} | {
            "model": self.model,
            "source": source,
            "table_id": table_id,
            "data_version": data_version,
            "checkpoint_step": self.checkpoint_step,
            "git_commit": self.git_commit,
        }
        # Opening and closing per row writes it to disk immediately.
        # Missing keys and None become empty cells.
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS, restval="").writerow(full)

        if self.predictions_dir is not None and proba is not None and table is not None:
            path = self.predictions_dir / source / f"{table_id}_split{row['split']}.npz"
            save_predictions(path, table, proba)


def save_predictions(path: str | Path, table: TableSample, proba: np.ndarray) -> None:
    """Save a table's test labels and predicted probabilities (see module doc)."""
    names = table.class_names or tuple(str(k) for k in range(proba.shape[1]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_true=table.y_test.numpy(),
        proba=np.asarray(proba, dtype=np.float32),
        class_names=np.array(names),
    )


def load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    """y_true, proba and class_names of one table, e.g. for a confusion matrix:

    p = load_predictions(path)
    ok = p["y_true"] >= 0
    confusion_matrix(p["y_true"][ok], p["proba"][ok].argmax(1))
    """
    with np.load(path) as data:
        return {key: data[key] for key in ("y_true", "proba", "class_names")}


def load_results(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Read one or many result CSVs (glob patterns allowed) into one DataFrame.

    Raises ValueError if a file's columns differ from COLUMNS.
    """
    if isinstance(paths, str | Path):
        paths = [paths]
    files = []
    for pattern in paths:
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            raise FileNotFoundError(f"No results file matches {pattern}")
        files += matches
    if not files:
        raise FileNotFoundError("No results file given")

    frames = []
    for file in files:
        df = pd.read_csv(file, dtype=DTYPES)
        missing = [c for c in COLUMNS if c not in df.columns]
        unknown = [c for c in df.columns if c not in COLUMNS]
        if missing or unknown:
            raise ValueError(
                f"{file}: missing columns {missing}, unknown columns {unknown}"
            )
        frames.append(df[list(COLUMNS)])
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per model and source: mean, std and mean rank of each metric.

    1. Only tables (and splits) that every model scored are kept, so all
       models are compared on the same tables.
    2. The splits of a table are averaged, so every table counts once.
    3. Across tables: mean and std of each metric, and the mean rank
       (1 = best model on that table). Ranks don't depend on how hard a
       table is, unlike the mean.
    A model at each checkpoint counts as its own model. NaN metrics (e.g.
    ROC AUC with one class in the test rows) are left out.
    """
    metrics = list(metric_names())
    ok = df[df["status"] == "ok"]
    # dropna=False keeps the baselines, which have an empty checkpoint_step.
    run = ["model", "checkpoint_step"]
    ok = ok.assign(run=ok.groupby(run, dropna=False).ngroup())
    runs_per_table = ok.groupby(TABLE_KEY)["run"].transform("nunique")
    runs_per_source = ok.groupby("source")["run"].transform("nunique")
    common = ok[runs_per_table == runs_per_source]
    if common.duplicated(["run", *TABLE_KEY]).any():
        raise ValueError("A model has two rows for the same table; check the CSVs")

    per_table = (
        common.groupby([*run, "source", "table_id"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    ranks = {
        f"{m}_rank": per_table.groupby(["source", "table_id"])[m].rank(
            ascending=not METRIC_REGISTRY[m].higher_is_better
        )
        for m in metrics
    }
    per_table = per_table.assign(**ranks)

    groups = per_table.groupby([*run, "source"], dropna=False)
    summary = groups[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    mean_ranks = groups[list(ranks)].mean()
    mean_ranks.columns = [f"{m}_mean_rank" for m in metrics]
    summary = summary.join(mean_ranks)
    summary.insert(0, "n_tables", groups.size())
    return summary.reset_index()


def write_summary(out: Path) -> pd.DataFrame:
    """Print the summary of every results CSV in out and save out/summary.txt."""
    summary = summarize(load_results(out / "*.csv"))
    if summary.empty:
        raise ValueError("No table was scored by every model")
    text = summary.to_string(index=False, float_format="%.3f")
    # .txt, not .csv: the folder's *.csv files must all be result files.
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    return summary
