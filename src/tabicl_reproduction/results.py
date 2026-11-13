"""Shared metrics and one results CSV format for every model.

Baselines, our TabICLv2 model and any later model are scored by the same
code and saved in the same format, so their results can be loaded together
and compared fairly.

Usage:

    from tabicl_reproduction.results import ResultsWriter, load_results

    writer = ResultsWriter("results/random_forest.csv", model="random_forest")
    for table in tables:
        proba = ...  # any model, shape [n_test, n_classes]
        writer.add(table, proba, source="tabpfn_val", table_id="31",
                   split_seed=0, time_s=elapsed)
        # if the model could not be fitted on this table:
        # writer.skip(table, "one class in training", source="tabpfn_val",
        #             table_id="31", split_seed=0)

    df = load_results("results/*.csv")  # all models in one DataFrame

`proba` holds the predicted probabilities for ALL test rows, one column per
class seen in training (n_classes = y_train.max() + 1). Test rows labeled -1
(their class never appears in training) can't be predicted and are left out
of the metrics; n_test counts only the rows that are scored.

One row per (model, table, split). Each add() or skip() appends its row
right away, so nothing is lost if a run crashes or hits a time limit.

| Column | Meaning | Example |
|---|---|---|
| model | Model name | random_forest |
| source | Where the table comes from | prior_val, tabpfn_val |
| table_id | OpenML id or prior table id | 31, prior_0017 |
| split_seed | Which train/test split | 0 |
| data_version | Version of the saved prior tables (empty for OpenML) | v1 |
| checkpoint_step | Training step of the checkpoint (empty for baselines) | 12000 |
| n_train, n_test, n_features, n_classes | Table size | 800, 200, 20, 2 |
| accuracy, balanced_accuracy, log_loss, roc_auc, ece | Metrics | 0.74 |
| time_s | Time for fit and predict on this table | 0.31 |
| status | ok or skipped | ok |
| error | Why it was skipped (empty if ok) | one class in training |
| git_commit | Code version | 827e7f6 |

log_loss clips probabilities at 1e-15, because baselines like random forests
can give exactly 0 to the true class. One row then adds at most
-log(1e-15) ≈ 34.5. The training loss (cross-entropy on logits) has no such
cap, so the two differ only for extremely confident wrong predictions.

An undefined metric (ROC AUC when the test rows hold only one class) is NaN.
A skipped table has empty metric cells, so every model has the same rows.
"""

import csv
import glob
import math
import subprocess
import warnings
from collections.abc import Hashable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from tabicl_reproduction.data import TableSample

METRICS = ("accuracy", "balanced_accuracy", "log_loss", "roc_auc", "ece")
COLUMNS = (
    "model",
    "source",
    "table_id",
    "split_seed",
    "data_version",
    "checkpoint_step",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    *METRICS,
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
INT_COLUMNS = ("split_seed", "n_train", "n_test", "n_features", "n_classes")
# Fixed types, so a file without rows can't turn numbers into text when
# files are combined. All other columns are floats (empty cells become NaN).
DTYPES: dict[Hashable, str] = {
    c: "str" if c in TEXT_COLUMNS else "int64" if c in INT_COLUMNS else "float64"
    for c in COLUMNS
}


def roc_auc(probs: np.ndarray, target: np.ndarray) -> float:
    """One-vs-rest ROC AUC over the classes present in the test rows."""
    present = np.unique(target)
    if len(present) < 2:
        return math.nan  # undefined with a single class
    # Renormalize over the present classes, as roc_auc_score requires.
    probs = probs[:, present]
    # A row with no probability on any present class (possible for k-NN or
    # random forests) carries no ranking information: treat it as uniform.
    probs[probs.sum(axis=1) == 0] = 1.0
    probs = probs / probs.sum(axis=1, keepdims=True)
    if len(present) == 2:
        return float(roc_auc_score(target == present[1], probs[:, 1]))
    return float(roc_auc_score(target, probs, multi_class="ovr", labels=present))


def expected_calibration_error(
    probs: np.ndarray, target: np.ndarray, n_bins: int = 15
) -> float:
    """How far predicted confidence is from actual accuracy (0 = calibrated)."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == target
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
        ece += gap * in_bin.mean()
    return float(ece)


def compute_metrics(y_true, proba) -> dict[str, float]:
    """The five metrics for one table; rows labeled -1 are left out.

    y_true: [n_test] true labels; proba: [n_test, n_classes] probabilities.
    """
    y_true = _to_numpy(y_true).astype(int)  # labels may come in as floats
    proba = _to_numpy(proba).astype(float)
    valid = y_true >= 0  # -1: class unseen in training, can't be predicted
    y, p = y_true[valid], proba[valid]
    if len(y) == 0:
        raise ValueError("No test row with a class seen in training")

    pred = p.argmax(axis=1)
    with warnings.catch_warnings():  # classes predicted but absent in y
        warnings.simplefilter("ignore", UserWarning)
        balanced = float(balanced_accuracy_score(y, pred))
    # Clip so a probability of exactly 0 on the true class stays finite.
    p_true = np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": balanced,
        "log_loss": float(-np.log(p_true).mean()),
        "roc_auc": roc_auc(p, y),
        "ece": expected_calibration_error(p, y),
    }


class ResultsWriter:
    """Appends one result row per (model, table, split) to a CSV file."""

    def __init__(
        self, path: str | Path, model: str, checkpoint_step: int | None = None
    ):
        self.path = Path(path)
        self.model = model
        self.checkpoint_step = checkpoint_step
        self.git_commit = _git_commit()

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
        table: TableSample,
        proba,
        *,
        source: str,
        table_id: str,
        split_seed: int,
        time_s: float,
        data_version: str | None = None,
    ) -> None:
        """Score the model's probabilities for this table and save the row."""
        proba = _to_numpy(proba)
        n_test, n_classes = len(table.y_test), int(table.y_train.max()) + 1
        if proba.shape != (n_test, n_classes):
            raise ValueError(
                f"proba must have shape ({n_test}, {n_classes}) "
                f"(test rows, training classes), got {proba.shape}"
            )
        if not (table.y_test >= 0).any():
            reason = "no test row with a class seen in training"
            self.skip(
                table,
                reason,
                source=source,
                table_id=table_id,
                split_seed=split_seed,
                data_version=data_version,
            )
            return

        row = self._row(table, source, table_id, split_seed, data_version)
        row |= compute_metrics(table.y_test, proba)
        row |= {"time_s": time_s, "status": "ok"}
        self._append(row)

    def skip(
        self,
        table: TableSample,
        reason: str,
        *,
        source: str,
        table_id: str,
        split_seed: int,
        data_version: str | None = None,
    ) -> None:
        """Save a row for a table the model could not handle, with the reason."""
        row = self._row(table, source, table_id, split_seed, data_version)
        row |= {"status": "skipped", "error": reason}
        self._append(row)

    def _row(self, table, source, table_id, split_seed, data_version) -> dict:
        return {
            "model": self.model,
            "source": source,
            "table_id": table_id,
            "split_seed": split_seed,
            "data_version": data_version,
            "checkpoint_step": self.checkpoint_step,
            "n_train": len(table.y_train),
            "n_test": int((table.y_test >= 0).sum()),
            "n_features": table.X_train.shape[1],
            "n_classes": int(table.y_train.max()) + 1,
            "git_commit": self.git_commit,
        }

    def _append(self, row: dict) -> None:
        # Opening and closing per row writes it to disk immediately.
        # Missing keys and None become empty cells.
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS, restval="").writerow(row)


def load_results(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Read one or many result CSVs (glob patterns allowed) into one DataFrame.

    Empty cells become NaN. Raises ValueError if a file's columns differ
    from COLUMNS.
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


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"  # e.g. in a container without git
