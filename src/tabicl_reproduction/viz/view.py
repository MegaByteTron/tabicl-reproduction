"""How a table's rows are placed on a 2D plot."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from sklearn.base import TransformerMixin, clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def to_numpy(t) -> np.ndarray:
    """Tensor (any device) or array-like -> numpy array; handy in overlays."""
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


@dataclass(frozen=True)
class View:
    """How a table's features are mapped onto the two plot axes.

    kind is "raw" (the table has exactly 2 features, plotted as they are),
    "projected" (more features, reduced to 2 with a scikit-learn method such
    as PCA) or "1d" (only one value per row, spread vertically by random
    jitter so overlapping points stay visible).
    """

    kind: Literal["raw", "projected", "1d"]
    xlabel: str
    ylabel: str
    # Fitted scaler + reduction method; None for "raw" and plain "1d".
    projection: Pipeline | None = None
    # Plotted range (low, high), each of shape [2]; set by show() before
    # overlays are drawn, so they can cover exactly the visible area.
    limits: tuple[np.ndarray, np.ndarray] | None = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:  # e.g. a table without test rows
            return np.empty((0, 2))
        P = X if self.projection is None else self.projection.transform(X)[:, :2]
        if P.shape[1] == 1:
            jitter = np.random.default_rng(0).uniform(-1, 1, size=len(P))
            P = np.column_stack([P[:, 0], jitter])
        return P


def make_view(
    X_train: np.ndarray, y_train: np.ndarray, projection: TransformerMixin
) -> View:
    """Choose how to plot a table, fitting the projection on training rows only."""
    n_features = X_train.shape[1]
    if n_features == 1:
        return View("1d", "feature 1", "(random jitter)")
    if n_features == 2:
        return View("raw", "feature 1", "feature 2")

    # Standardize first, so features with large values don't dominate.
    pipeline = make_pipeline(StandardScaler(), clone(projection))
    pipeline.fit(X_train, y_train)  # supervised methods like LDA use y
    method = pipeline[-1]
    name = _short_name(method)
    n_out = min(pipeline.transform(X_train[:1]).shape[1], 2)

    ratios = getattr(method, "explained_variance_ratio_", None)
    labels = [
        f"{name} {i + 1}" + (f" ({ratios[i]:.0%} var.)" if ratios is not None else "")
        for i in range(n_out)
    ]
    if n_out == 1:  # e.g. LDA on a 2-class table gives only one axis
        return View("1d", labels[0], "(random jitter)", projection=pipeline)
    return View("projected", labels[0], labels[1], projection=pipeline)


def _short_name(method) -> str:
    """PCA -> "PC", LinearDiscriminantAnalysis -> "LD", Isomap -> "Isomap"."""
    known = {"PCA": "PC", "LinearDiscriminantAnalysis": "LD", "KernelPCA": "KPC"}
    name = type(method).__name__
    return known.get(name, name)
