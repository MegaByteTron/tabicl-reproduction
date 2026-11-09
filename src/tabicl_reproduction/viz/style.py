"""Colors, markers and matplotlib settings shared by all figures.

Classes are told apart by color AND marker shape, so plots stay readable
for colorblind readers and in grayscale print. The eight colors are a
colorblind-checked categorical palette, used in this fixed order.
"""

from typing import Any

from matplotlib.typing import RcKeyType

CLASS_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
CLASS_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "p")

INK = "#0b0b0b"  # titles
INK_SECONDARY = "#52514e"  # subtitles, legend text
INK_MUTED = "#898781"  # axis labels
AXIS = "#c3c2b7"  # axis lines
UNSEEN = "#b5b3ab"  # test rows whose class never appears in training

RC_PARAMS: dict[RcKeyType, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelsize": 7,
    "axes.labelcolor": INK_MUTED,
    "xtick.bottom": False,
    "ytick.left": False,
    "xtick.labelbottom": False,
    "ytick.labelleft": False,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def class_style(label: int) -> tuple[str, str]:
    """Color and marker of a class; classes 8 and 9 reuse colors with new shapes.

    From class 10 on, colors and shapes repeat, so tables with many classes
    still plot instead of crashing.
    """
    color = CLASS_COLORS[label % len(CLASS_COLORS)]
    return color, CLASS_MARKERS[label % len(CLASS_MARKERS)]
