"""Confusion matrices of a few tables, our model and a baseline side by side.

Classes differ between tables, so matrices can't be pooled over tables, and
the results CSVs hold no predictions: they come from fresh predictions
(see make_figures.py).
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from style import COLORS, TEXT_WIDTH_IN, model_label, save, use_style


@dataclass
class ConfusionCase:
    """One table: true labels and each model's predicted labels (ints 0..k-1)."""

    title: str
    y_true: np.ndarray
    y_pred: dict[str, np.ndarray]  # model name -> predictions


def plot_confusion_matrices(cases: list[ConfusionCase], outfile: Path) -> None:
    """One row per table, one column per model; rows normalized per true class.

    A cell is the fraction of a true class predicted as each class, so the
    diagonal is the recall of each class.
    """
    n_models = max(len(case.y_pred) for case in cases)
    use_style()
    fig, axes = plt.subplots(
        len(cases),
        n_models,
        figsize=(TEXT_WIDTH_IN, 2.3 * len(cases)),
        squeeze=False,
    )
    cmap = sns.light_palette(COLORS[0], as_cmap=True)
    for row, case in zip(axes, cases, strict=True):
        # Every class that is true or predicted, so all matrices of a table match.
        n_classes = max(int(y.max()) for y in [case.y_true, *case.y_pred.values()])
        labels = np.arange(n_classes + 1)
        for ax, (model, y_pred) in zip(row, case.y_pred.items(), strict=False):
            matrix = confusion_matrix(
                case.y_true, y_pred, labels=labels, normalize="true"
            )
            with warnings.catch_warnings():  # classes predicted but absent in y_true
                warnings.simplefilter("ignore", UserWarning)
                balanced = balanced_accuracy_score(case.y_true, y_pred)
            sns.heatmap(
                np.nan_to_num(matrix),  # a class without test rows: all 0
                ax=ax,
                cmap=cmap,
                vmin=0,
                vmax=1,
                square=True,
                annot=len(labels) <= 6,  # numbers get unreadable beyond that
                fmt=".2f",
                annot_kws={"fontsize": 6},
                cbar=False,
                linewidths=0.5,
                linecolor="white",
            )
            ax.set(
                xlabel="Predicted class",
                ylabel="True class",
                title=f"{case.title}\n{model_label(model)} "
                f"(balanced acc. {balanced:.2f})",
            )
            ax.grid(False)
        for ax in row[len(case.y_pred) :]:
            ax.set_visible(False)
    fig.tight_layout()
    save(fig, outfile)
