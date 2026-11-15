"""Figures comparing all models on the result tables (prior_val and tabpfn_val).

Every function takes the DataFrame from tabicl_reproduction.results.load_results
and an outfile. As in baselines.summarize, only tables that every model scored
are compared, and the two splits of an OpenML table are averaged first, so
each table counts once.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from style import (
    COLORS,
    INK_MUTED,
    SOURCE_LABELS,
    TEXT_WIDTH_IN,
    is_ours,
    model_color,
    model_label,
    model_order,
    save,
    use_style,
)

from tabicl_reproduction.data import TABPFN_VAL_DATASETS

SOURCES = ("prior_val", "tabpfn_val")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "log_loss": "Log loss",
    "roc_auc": "ROC AUC",
    "ece": "ECE",
}
LOWER_IS_BETTER = ("log_loss", "ece")
DATASET_NAMES = {str(i): name for name, i in TABPFN_VAL_DATASETS.items()}


def dataset_label(table_id: str) -> str:
    """OpenML dataset name in typewriter font (cmr10 has no underscore)."""
    name = DATASET_NAMES.get(table_id, table_id).replace("_", r"\_")
    return rf"$\mathtt{{{name}}}$"


# Bins of the table properties, (upper edges, labels); a bin holds the values
# above the previous edge up to its own edge. cmr10 has no en dash and no ">",
# hence "-" and "$>$".
CATEGORIES = {
    "n_classes": ("Classes", [2, 5, np.inf], ["2", "3-5", "6-10"]),
    "n_features": ("Features", [10, 30, np.inf], [r"$\leq$10", "11-30", "$>$30"]),
    "n_train": (
        "Training rows",
        [500, 1500, np.inf],
        [r"$\leq$500", "501-1500", "$>$1500"],
    ),
}


def per_table(df: pd.DataFrame, source: str, metric: str = "roc_auc") -> pd.DataFrame:
    """One row per table of source, one column per model (in model_order).

    Splits are averaged; tables that a model skipped or where the metric is
    undefined (NaN) are dropped, so all models share the same tables.
    """
    ok = df[(df["source"] == source) & (df["status"] == "ok")]
    wide = ok.pivot_table(index="table_id", columns="model", values=metric)
    return wide.dropna()[model_order(wide.columns)]


def table_info(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """n_train, n_features, n_classes and n_test per table (from split 0)."""
    rows = df[df["source"] == source].sort_values("split_seed")
    return rows.groupby("table_id")[[*CATEGORIES, "n_test"]].first()


def best_baseline(scores: pd.DataFrame) -> str:
    """The baseline with the highest mean in a per_table() frame."""
    return scores[[m for m in scores.columns if not is_ours(m)]].mean().idxmax()


def mean_ci(scores: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Mean per column and half width of its 95% confidence interval."""
    return scores.mean(), 1.96 * scores.std() / np.sqrt(len(scores))


def _dot_panels(stats: dict[str, pd.DataFrame], xlabel: str, outfile: Path) -> None:
    """One panel per source: a dot with an error bar per model, ours on top."""
    use_style()
    fig, axes = plt.subplots(1, len(stats), figsize=(TEXT_WIDTH_IN, 1.7))
    axes = np.atleast_1d(axes)
    for ax, (source, scores) in zip(axes, stats.items(), strict=True):
        mean, ci = mean_ci(scores)
        models = list(scores.columns)
        y = np.arange(len(models))[::-1]  # first model at the top
        for yi, model in zip(y, models, strict=True):
            ax.errorbar(
                mean[model],
                yi,
                xerr=ci[model],
                fmt="o",
                markersize=4,
                capsize=2,
                color=model_color(model),
            )
            ax.annotate(
                f"{mean[model]:.3f}",
                (mean[model], yi),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=6,
            )
        ax.set_yticks(y, [model_label(m) for m in models])
        ax.set_ylim(y.min() - 0.6, y.max() + 0.8)
        ax.grid(axis="y", visible=False)
        ax.set(
            xlabel=xlabel,
            title=f"{SOURCE_LABELS.get(source, source)}, {len(scores)} tables",
        )
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)  # same models in the same order
    fig.tight_layout()
    save(fig, outfile)


def plot_overall(df: pd.DataFrame, outfile: Path, metric: str = "roc_auc") -> None:
    """Mean ROC AUC per model with 95% confidence intervals of the mean,
    one panel per source (synthetic prior tables, real OpenML tables)."""
    stats = {s: per_table(df, s, metric) for s in SOURCES if (df["source"] == s).any()}
    _dot_panels(stats, f"Mean {METRIC_LABELS[metric]}", outfile)


def plot_mean_rank(df: pd.DataFrame, outfile: Path, metric: str = "roc_auc") -> None:
    """Mean rank per model (1 = best on a table) with 95% confidence intervals.

    As the critical-difference diagrams of TabICL/TabICLv2: ranks don't
    depend on how easy a table is, unlike mean scores.
    """
    stats = {
        s: per_table(df, s, metric).rank(axis=1, ascending=metric in LOWER_IS_BETTER)
        for s in SOURCES
        if (df["source"] == s).any()
    }
    _dot_panels(stats, "Mean rank (lower is better)", outfile)


def plot_per_dataset(df: pd.DataFrame, outfile: Path, ours: str = "tabicl_best"):
    """Our ROC AUC vs the best baseline's per table, one panel per source.

    Points above the diagonal are tables we win. Color: binary or multiclass.
    On OpenML, the tables we lose most on are labeled by dataset name.
    """
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.75))
    for ax, source in zip(axes, SOURCES, strict=True):
        scores = per_table(df, source)
        if ours not in scores or scores.empty:
            ax.set_visible(False)
            continue
        baseline = best_baseline(scores)
        x, y = scores[baseline], scores[ours]
        binary = table_info(df, source).loc[scores.index, "n_classes"] == 2
        small = source == "prior_val"  # many points: smaller and lighter
        for mask, color, label in [
            (binary, COLORS[0], "binary"),
            (~binary, COLORS[1], "multiclass"),
        ]:
            ax.scatter(
                x[mask],
                y[mask],
                s=4 if small else 12,
                alpha=0.4 if small else 0.8,
                color=color,
                edgecolor="none",
                label=label,
            )
        low = min(x.min(), y.min()) - 0.02
        ax.plot([low, 1], [low, 1], color=INK_MUTED, linestyle="--", linewidth=0.8)
        ax.set(xlim=(low, 1.01), ylim=(low, 1.01), aspect="equal")
        ax.set(
            xlabel=f"ROC AUC, {model_label(baseline)}",
            ylabel=f"ROC AUC, {model_label(ours)}",
        )
        wins = int((y > x).sum())
        ax.set_title(f"{SOURCE_LABELS[source]}: ours better on {wins}/{len(x)}")
        if not small:
            # Labels stacked in the free lower right corner, arrows to the points.
            for i, table_id in enumerate((y - x).nsmallest(3).index):
                ax.annotate(
                    dataset_label(table_id),
                    (x[table_id], y[table_id]),
                    xytext=(0.98, 0.04 + 0.08 * i),
                    textcoords="axes fraction",
                    ha="right",
                    fontsize=6,
                    arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 0.5},
                )
        legend = ax.legend(loc="upper left", markerscale=2 if small else 1)
        for handle in legend.legend_handles:
            handle.set_alpha(1)
    fig.tight_layout()
    save(fig, outfile)


def plot_by_category(df: pd.DataFrame, outfile: Path, metric: str = "roc_auc"):
    """Mean ROC AUC per model in bins of n_classes, n_features and n_train.

    One row per source; the number of tables per bin is under each tick.
    """
    sources = [s for s in SOURCES if (df["source"] == s).any()]
    use_style()
    fig, axes = plt.subplots(
        len(sources),
        len(CATEGORIES),
        figsize=(TEXT_WIDTH_IN, 1.75 * len(sources) + 0.3),
        sharey="row",
        squeeze=False,
    )
    for row, source in zip(axes, sources, strict=True):
        scores = per_table(df, source, metric)
        info = table_info(df, source).loc[scores.index]
        for ax, (column, (title, edges, labels)) in zip(
            row, CATEGORIES.items(), strict=True
        ):
            bins = pd.cut(info[column], [-np.inf, *edges], labels=labels)
            means = scores.groupby(bins, observed=False).mean()
            counts = bins.value_counts().reindex(labels)
            x = np.arange(len(labels))
            for model in scores.columns:
                ax.plot(
                    x,
                    means[model],
                    marker="o",
                    markersize=3,
                    color=model_color(model),
                    label=model_label(model),
                    linewidth=1.6 if is_ours(model) else 0.9,
                    zorder=3 if is_ours(model) else 2,
                )
            ax.set_xticks(
                x, [f"{b}\n($n$={c})" for b, c in zip(labels, counts, strict=True)]
            )
            ax.set_xlim(-0.3, len(labels) - 0.7)
            ax.set_xlabel(title)
        row[0].set_ylabel(f"{METRIC_LABELS[metric]}\n{SOURCE_LABELS[source]}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, outfile)
