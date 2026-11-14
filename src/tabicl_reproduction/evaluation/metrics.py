"""The metrics every model is scored with, during pretraining and after.

Each metric is registered with @register_metric and computed from the true
labels and the predicted probabilities:

    @register_metric(higher_is_better=True)
    def accuracy(y_true, proba) -> float: ...

    evaluate_metrics(y_true, proba)  # {"accuracy": ..., "log_loss": ..., ...}

The probabilities come from one predict_proba call per table, shared by all
metrics (and saved for confusion matrices). y_true holds labels
0..n_classes-1, matching the proba columns; -1 marks a test row whose class
never appears in training. evaluate_metrics leaves those rows out once, for
every metric, because no model can predict such a class.
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

type MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Metric:
    fn: MetricFn
    higher_is_better: bool


METRIC_REGISTRY: dict[str, Metric] = {}


def register_metric(
    name: str | None = None, *, higher_is_better: bool = True
) -> Callable[[MetricFn], MetricFn]:
    """Add a metric fn(y_true, proba) -> float to every evaluation."""

    def decorator(fn: MetricFn) -> MetricFn:
        metric_name = name or getattr(fn, "__name__", repr(fn))
        if metric_name in METRIC_REGISTRY:
            raise ValueError(f"Metric '{metric_name}' already registered")
        METRIC_REGISTRY[metric_name] = Metric(fn, higher_is_better)
        return fn

    return decorator


@register_metric()
def accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float((proba.argmax(axis=1) == y_true).mean())


@register_metric()
def balanced_accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    with warnings.catch_warnings():  # classes predicted but absent in y_true
        warnings.simplefilter("ignore", UserWarning)
        return float(balanced_accuracy_score(y_true, proba.argmax(axis=1)))


@register_metric()
def f1_macro(y_true: np.ndarray, proba: np.ndarray) -> float:
    # Only the classes present in the test rows, like balanced accuracy.
    labels = np.unique(y_true)
    return float(
        f1_score(
            y_true,
            proba.argmax(axis=1),
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


@register_metric(higher_is_better=False)
def log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean -log p(true class), clipped at 1e-15 (at most 34.5 per row).

    Baselines can give exactly 0 to the true class; without the clip one
    such row would make the loss infinite.
    """
    p_true = np.clip(proba[np.arange(len(y_true)), y_true], 1e-15, 1.0)
    return float(-np.log(p_true).mean())


@register_metric()
def roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """One-vs-rest ROC AUC over the classes present in the test rows.

    NaN with a single class in the test rows (undefined).
    """
    present = np.unique(y_true)
    if len(present) < 2:
        return math.nan
    # Renormalize over the present classes, as roc_auc_score requires.
    proba = proba[:, present]
    # A row with no probability on any present class (possible for k-NN or
    # random forests) carries no ranking information: treat it as uniform.
    proba[proba.sum(axis=1) == 0] = 1.0
    proba = proba / proba.sum(axis=1, keepdims=True)
    if len(present) == 2:
        return float(roc_auc_score(y_true == present[1], proba[:, 1]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=present))


@register_metric("ece", higher_is_better=False)
def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15
) -> float:
    """How far predicted confidence is from actual accuracy (0 = calibrated)."""
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
        ece += gap * in_bin.mean()
    return float(ece)


def evaluate_metrics(y_true, proba) -> dict[str, float]:
    """Every registered metric for one table; rows labeled -1 are left out.

    y_true: [n_test] labels 0..n_classes-1 (-1: class unseen in training).
    proba: [n_test, n_classes] probabilities, column k for label k.
    """
    y_true = to_numpy(y_true).astype(int)  # labels may come in as floats
    proba = to_numpy(proba).astype(float)
    valid = y_true >= 0
    if not valid.any():
        raise ValueError("No test row with a class seen in training")
    y, p = y_true[valid], proba[valid]
    # .copy(): a metric must not change the probabilities the next one sees.
    return {name: metric.fn(y, p.copy()) for name, metric in METRIC_REGISTRY.items()}


def metric_names() -> tuple[str, ...]:
    return tuple(METRIC_REGISTRY)


def to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
