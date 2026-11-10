"""Overlays: extra content drawn on a show() panel, e.g. decision boundaries."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from sklearn.base import BaseEstimator, clone, is_classifier

from tabicl_reproduction.data.table import TableSample

from .style import class_style
from .view import View, to_numpy


class Overlay(Protocol):
    """Anything that draws extra content on one panel of show().

    Called once per panel, after the axis limits are set and before the
    points are drawn. Any function or object with this call signature works;
    no base class is needed. Use view.transform() to map feature rows to plot
    coordinates and view.limits for the visible area. Set zorder > 4 to draw
    above the points.

    Example:
        def class_means(ax, sample, view):
            ...

        show(tables, overlays=[class_means, DecisionBoundary(classifier)])
    """

    def __call__(self, ax: Axes, sample: TableSample, view: View, /) -> None: ...


@dataclass(frozen=True)
class DecisionBoundary:
    """Shade each region by the predicted class; stronger = more confident.

    Only for tables with exactly two features: in a reduced view (e.g.
    PCA), one point on the plot stands for many different rows, so there is
    no single prediction to draw.

    The classifier must be a scikit-learn classifier with predict_proba
    (our own Classifier counts if it inherits ClassifierMixin and
    BaseEstimator).
    A fresh copy is fitted on each table's training rows, so the classifier
    you pass in stays unchanged.

    Example:
        show(samples, overlays=[DecisionBoundary(KNeighborsClassifier())])
    """

    classifier: object
    resolution: int = 150  # grid points per axis
    max_alpha: float = 0.35  # opacity at full confidence

    def __post_init__(self):
        # Fail here, not halfway through drawing.
        name = type(self.classifier).__name__
        # is_classifier() alone crashes on non-scikit-learn objects (e.g. torch).
        is_sklearn = isinstance(self.classifier, BaseEstimator)
        if not (is_sklearn and is_classifier(self.classifier)):
            raise TypeError(
                f"DecisionBoundary needs a scikit-learn classifier, got {name}"
            )
        # e.g. SVC() only has predict_proba with probability=True
        if not hasattr(self.classifier, "predict_proba"):
            raise TypeError(f"{name} has no predict_proba")

    def __call__(self, ax: Axes, sample: TableSample, view: View) -> None:
        if view.kind != "raw":
            raise ValueError(
                "DecisionBoundary needs tables with exactly 2 features, "
                f"got {sample.X_train.shape[1]}"
            )
        X_train, y_train = to_numpy(sample.X_train), to_numpy(sample.y_train)
        assert view.limits is not None, "show() sets limits before overlays"
        low, high = view.limits  # the visible area of the panel
        xs = np.linspace(low[0], high[0], self.resolution)
        ys = np.linspace(low[1], high[1], self.resolution)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

        fitted = clone(self.classifier).fit(X_train, y_train)
        # predict_proba only has columns for classes seen in training;
        # place them at their label index so colors match the points.
        n_classes = int(y_train.max()) + 1
        proba = np.zeros((len(grid), n_classes))
        proba[:, fitted.classes_] = fitted.predict_proba(grid)
        winner = proba.argmax(axis=1)
        # 0 when all classes are equally likely, 1 when one is certain.
        # With a single class (only label 0 in training) it is always certain.
        chance = 1 / n_classes
        confidence = (
            (proba.max(axis=1) - chance) / (1 - chance)
            if n_classes > 1
            else np.ones(len(grid))
        )

        colors = np.array([to_rgb(class_style(c)[0]) for c in range(n_classes)])
        image = np.concatenate(
            [colors[winner], (self.max_alpha * confidence)[:, None]], axis=1
        ).reshape(self.resolution, self.resolution, 4)

        ax.imshow(
            image,
            origin="lower",
            extent=(low[0], high[0], low[1], high[1]),
            aspect="auto",
            interpolation="bilinear",
            zorder=0,
        )
