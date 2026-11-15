"""Shared look of all report figures, so they read as one set.

Colors come from the repo's colorblind-checked palette. Figures are sized
to the report's text width, so fonts appear in the PDF at the size set here
(no scaling), and saved as vector PDFs into report/figures/.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.typing import RcKeyType

from tabicl_reproduction.viz.style import CLASS_COLORS, INK, INK_MUTED

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# \linewidth of the report in inches (print \the\linewidth in LaTeX to check).
TEXT_WIDTH_IN = 5.3

# One color per method/run, in this order, everywhere in the report.
COLORS = CLASS_COLORS

# Display names of the models in the results CSVs; our model first, so it
# always gets the first color. Unknown models keep their CSV name.
MODEL_LABELS = {
    "tabicl_best": "Ours",
    "hist_gradient_boosting": "Gradient boosting",
    "random_forest": "Random forest",
    "logistic_regression": "Logistic regression",
    "knn": "$k$-NN",
    "tabicl_model": "Ours (last step)",
}
SOURCE_LABELS = {
    "prior_val": r"Prior ($\mathit{prior\_val}$)",
    "tabpfn_val": r"OpenML ($\mathit{tabpfn\_val}$)",
}


def model_order(models) -> list[str]:
    """Models in the order of MODEL_LABELS, unknown ones after, sorted."""
    known = [m for m in MODEL_LABELS if m in set(models)]
    return known + sorted(set(models) - set(known))


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def model_color(model: str) -> str:
    """The same color for a model in every figure (gray if unknown)."""
    names = list(MODEL_LABELS)
    return COLORS[names.index(model)] if model in names else INK_MUTED


def is_ours(model: str) -> bool:
    return model.startswith("tabicl")


RC_PARAMS: dict[RcKeyType, Any] = {
    # Computer Modern, the report's LaTeX font (ships with matplotlib), so
    # figure text matches the body text.
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # cmr10 has no minus sign otherwise
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": "#e6e5df",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed real fonts, so text stays selectable
}


def use_style() -> None:
    plt.rcParams.update(RC_PARAMS)
    # fontTools warns about cmr10's old timestamps when embedding it in a PDF.
    logging.getLogger("fontTools").setLevel(logging.ERROR)


def save(fig: plt.Figure, outfile: str | Path) -> Path:
    """Save to outfile (usually figures/<name>.pdf, included with
    \\includegraphics{<name>}) and close the figure."""
    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")
    return path
