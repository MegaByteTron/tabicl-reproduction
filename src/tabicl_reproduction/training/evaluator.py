import math
import time
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn

from tabicl_reproduction.data import TableSample, load_data, to_model_inputs

SUMMARY_METRICS = ("accuracy", "balanced_accuracy", "log_loss", "roc_auc", "ece")


def roc_auc(probs: np.ndarray, target: np.ndarray) -> float:
    """One-vs-rest ROC AUC over the classes present in the test rows."""
    present = np.unique(target)
    if len(present) < 2:
        return math.nan  # undefined with a single class
    # Renormalize over the present classes, as roc_auc_score requires.
    probs = probs[:, present]
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


class Evaluator:
    """Scores a model on real tables (default: the tabpfn_val benchmark)."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        tables: Iterable[TableSample] | None = None,
        max_classes: int = 10,
    ):
        self.model = model
        self.device = device
        self.tables = load_data("tabpfn_val") if tables is None else tables
        self.max_classes = max_classes

    def _score(self, sample: TableSample) -> dict[str, Any] | None:
        n_classes = int(sample.y_train.max()) + 1
        valid = sample.y_test >= 0  # test classes unseen in training can't be predicted
        if n_classes > self.max_classes or not valid.any():
            return None

        start = time.perf_counter()
        x, y = to_model_inputs(sample)
        logits = self.model(x.to(self.device), y.to(self.device))[0]
        # Only the classes that occur in training can be predicted.
        logits = logits[:, :n_classes].float().cpu()[valid]
        target = sample.y_test[valid]
        probs = torch.softmax(logits, dim=-1).numpy()
        pred = probs.argmax(axis=1)
        y_true = target.numpy()

        with warnings.catch_warnings():  # classes predicted but absent in y_true
            warnings.simplefilter("ignore", UserWarning)
            balanced = float(balanced_accuracy_score(y_true, pred))
        return {
            "dataset_id": sample.dataset_id,
            "n_train": len(sample.y_train),
            "n_test": int(valid.sum()),
            "n_features": sample.X_train.shape[1],
            "n_classes": n_classes,
            "accuracy": float((pred == y_true).mean()),
            "balanced_accuracy": balanced,
            "log_loss": nn.functional.cross_entropy(logits, target).item(),
            "roc_auc": roc_auc(probs, y_true),
            "ece": expected_calibration_error(probs, y_true),
            "time_s": time.perf_counter() - start,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Per-dataset scores and their means.

        Returns {"summary": {metric: mean, ..., n_datasets, time_s},
        "datasets": [one dict per table]}. Tables with more classes than the
        model supports are skipped.
        """
        was_training = self.model.training
        self.model.eval()
        start = time.perf_counter()
        try:
            rows = [r for s in self.tables if (r := self._score(s)) is not None]
        finally:
            self.model.train(was_training)

        if not rows:
            raise ValueError("No table could be evaluated")
        summary = {m: float(np.nanmean([r[m] for r in rows])) for m in SUMMARY_METRICS}
        summary |= {"n_datasets": len(rows), "time_s": time.perf_counter() - start}
        return {"summary": summary, "datasets": rows}
