"""Shared look of all report figures, so they read as one set.

Colors come from the repo's colorblind-checked palette. Figures are sized
to the report's text width, so fonts appear in the PDF at the size set here
(no scaling), and saved as vector PDFs into report/figures/.
"""

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


def save(fig: plt.Figure, name: str) -> Path:
    """Save as figures/<name>.pdf; include it with \\includegraphics{<name>}."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")
    return path
