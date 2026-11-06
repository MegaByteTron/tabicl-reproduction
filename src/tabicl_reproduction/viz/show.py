import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA

from tabicl_reproduction.data.benchmarks import BENCHMARKS
from tabicl_reproduction.data.table import TableSample

from .overlays import Overlay
from .style import INK_SECONDARY, RC_PARAMS, UNSEEN, class_style
from .view import make_view, to_numpy

# Dataset id -> name, for panel titles.
DATASET_NAMES = {
    dataset_id: name
    for datasets in BENCHMARKS.values()
    for name, dataset_id in datasets.items()
}


def show(
    x: TableSample | Sequence[TableSample],
    outfile: str | Path | None = None,
    *,
    overlays: Sequence[Overlay] = (),
    projection: TransformerMixin | None = None,
    titles: Sequence[str] | None = None,
    ncols: int = 4,
    panel_size: float = 2.4,
    max_points: int = 400,
    robust_limits: bool = True,
) -> Figure:
    """Plot one or more tables as a grid of 2D scatter plots.

    Each panel shows one table: training rows as filled markers, test rows
    as hollow markers, one color and shape per class. Tables with more than
    two features are reduced to two dimensions with `projection`.

    Args:
        x: a sample (X_train, y_train, X_test, y_test) or a list of them.
        outfile: save the figure here; the format follows the extension
            (use .pdf for the report).
        overlays: drawn on every panel below the points, e.g.
            DecisionBoundary(...).
        projection: scikit-learn method that reduces tables with more than 2
            features to 2, fitted on each table's (standardized) training
            rows; only the first two output axes are plotted. Default PCA(2).
            Others: LinearDiscriminantAnalysis() uses the labels to separate
            classes; Isomap(n_components=2) or KernelPCA(2, kernel="rbf")
            follow curved structure. Needs a transform() method, so TSNE does not work.
        titles: one title per panel; default is the dataset name.
        ncols: panels per row.
        panel_size: width and height of one panel in inches.
        max_points: rows plotted per panel at most, to keep plots readable.
        robust_limits: limit each axis to at most 3 interquartile ranges
            beyond the middle half of the data (Tukey's "far out" fences),
            so a few outliers don't squash the rest; points outside are
            drawn at the edge. Set False to show the full range, e.g. when
            looking at outliers.

    Returns:
        The matplotlib figure, for further changes or plt.show().
    """
    samples = _as_samples(x)
    projection = PCA(n_components=2) if projection is None else projection
    if not hasattr(projection, "transform"):
        raise TypeError(
            f"projection needs a transform() method to place test rows, "
            f"{type(projection).__name__} has none; try PCA, LDA or Isomap"
        )
    n = len(samples)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    # Apply settings without affecting global Matplotlib configuration.
    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * panel_size, nrows * panel_size + 0.5),
            squeeze=False,  # Prevent Matplotlib from changing the axes array dimensions
            layout="constrained",
        )
        for i, (ax, sample) in enumerate(zip(axes.flat, samples, strict=False)):
            title = titles[i] if titles is not None else _default_title(sample, i)
            _draw_panel(
                ax, sample, title, overlays, projection, max_points, robust_limits
            )
        for ax in axes.flat[n:]:
            ax.set_axis_off()

        _add_legend(fig, samples)

        if outfile is not None:
            fig.savefig(outfile)
    return fig


def _as_samples(x) -> list[TableSample]:
    is_single = isinstance(x, TableSample) or (
        isinstance(x, tuple) and len(x) == 4 and isinstance(x[0], torch.Tensor)
    )
    items = [x] if is_single else list(x)
    if not items:
        raise ValueError("show() needs at least one table")
    return [s if isinstance(s, TableSample) else TableSample(*s) for s in items]


def _default_title(sample: TableSample, index: int) -> str:
    if sample.dataset_id is None:
        return f"Table {index + 1}"
    name = DATASET_NAMES.get(sample.dataset_id, f"OpenML {sample.dataset_id}")
    return name if len(name) <= 24 else name[:23] + "…"


def _subsample(n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, size=k, replace=False))


def _limits(points: np.ndarray, robust: bool) -> tuple[np.ndarray, np.ndarray]:
    """Axis range with 5% padding; robust = cut at Tukey's far-out fences."""
    low, high = points.min(axis=0), points.max(axis=0)
    if robust:
        q1, q3 = np.percentile(points, [25, 75], axis=0)
        iqr = q3 - q1
        low = np.maximum(low, q1 - 3 * iqr)
        high = np.minimum(high, q3 + 3 * iqr)
    span = np.where(high > low, high - low, 1.0)  # avoid a zero-width axis
    return low - 0.05 * span, high + 0.05 * span  # Add 5% padding on each side.


def _draw_panel(
    ax: Axes,
    sample: TableSample,
    title: str,
    overlays: Sequence[Overlay],
    projection: TransformerMixin,
    max_points: int,
    robust_limits: bool,
) -> None:
    X_train, y_train, X_test, y_test = (to_numpy(t) for t in sample)
    view = make_view(X_train, y_train, projection)

    # Split the point budget between train and test rows by their share.
    n_train_shown = round(max_points * len(X_train) / (len(X_train) + len(X_test)))
    tr = _subsample(len(X_train), max(n_train_shown, 1))
    te = _subsample(len(X_test), max(max_points - len(tr), 1))
    P_train, P_test = view.transform(X_train[tr]), view.transform(X_test[te])
    y_tr, y_te = y_train[tr], y_test[te]

    low, high = _limits(np.concatenate([P_train, P_test]), robust_limits)
    view = replace(view, limits=(low, high))
    P_train, P_test = np.clip(P_train, low, high), np.clip(P_test, low, high)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])

    for overlay in overlays:
        overlay(ax, sample, view)

    for label in np.unique(np.concatenate([y_tr, y_te])):
        if label < 0:
            continue
        color, marker = class_style(int(label))
        m = y_tr == label
        ax.scatter(
            *P_train[m].T,
            s=18,
            marker=marker,
            color=color,
            edgecolors="white",  # thin ring keeps overlapping points apart
            linewidths=0.4,
            zorder=3,
        )
        m = y_te == label
        ax.scatter(
            *P_test[m].T,
            s=18,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            zorder=4,
        )

    unseen = y_te < 0
    if unseen.any():
        ax.scatter(*P_test[unseen].T, s=18, marker="x", color=UNSEEN, zorder=4)

    ax.set_title(title, loc="left", pad=12)
    n_classes = len(np.unique(y_train))
    ax.text(
        0,
        1.02,
        f"{len(X_train)} train · {len(X_test)} test · "
        f"{X_train.shape[1]} feat. · {n_classes} cl.",
        transform=ax.transAxes,
        fontsize=7,
        color=INK_SECONDARY,
        va="bottom",
    )
    ax.set_xlabel(view.xlabel)
    ax.set_ylabel(view.ylabel)


def _add_legend(fig: Figure, samples: list[TableSample]) -> None:
    n_classes = max(int(to_numpy(s.y_train).max()) + 1 for s in samples)
    handles = []
    for label in range(n_classes):
        color, marker = class_style(label)
        handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                color=color,
                linestyle="",
                markeredgecolor="white",
                markersize=6,
                label=f"class {label}",
            )
        )
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            color=INK_SECONDARY,
            linestyle="",
            markersize=6,
            label="train (filled)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            markeredgecolor=INK_SECONDARY,
            linestyle="",
            markersize=6,
            label="test (hollow)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(len(handles), 6),
        labelcolor=INK_SECONDARY,
    )
