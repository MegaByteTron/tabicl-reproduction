"""The metrics every model is scored with, during pretraining and after.

This is the only implementation: the training evaluator, the experiment and
the tests all call compute_metrics().
"""

import math
import warnings

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

METRICS = ("accuracy", "balanced_accuracy", "log_loss", "roc_auc", "ece")
# For ranking models: True if a higher value is better.
HIGHER_IS_BETTER = {
    "accuracy": True,
    "balanced_accuracy": True,
    "log_loss": False,
    "roc_auc": True,
    "ece": False,
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

    y_true: [n_test] labels 0..n_classes-1 (-1: class unseen in training).
    proba: [n_test, n_classes] probabilities, column k for label k.
    log_loss clips each probability at 1e-15 (at most 34.5 per row), because
    baselines can give exactly 0 to the true class.
    """
    y_true = to_numpy(y_true).astype(int)  # labels may come in as floats
    proba = to_numpy(proba).astype(float)
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


def to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
