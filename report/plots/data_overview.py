"""Section 2 figures: example tables, and the shapes of all evaluation tables.

    uv run python -m tabicl_reproduction.explore   # writes report/data_stats.csv
    uv run python report/plots/data_overview.py

Writes report/figures/example_tables.pdf (show() on 3 prior and 3 OpenML
tables) and table_shapes.pdf (training rows vs. features, and classes, of the
prior tables and the OpenML tables).
"""

import argparse
import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.typing import RcKeyType
from style import (
    FIGURES_DIR,
    INK,
    INK_MUTED,
    RC_PARAMS,
    TEXT_WIDTH_IN,
    save,
    use_style,
)

from tabicl_reproduction.data import BENCHMARKS, OpenMLDataset, TableSample, load_data
from tabicl_reproduction.data.loader import default_data_dir
from tabicl_reproduction.prior_val import PRIOR_SETTINGS
from tabicl_reproduction.viz import show

STATS = Path(__file__).resolve().parents[1] / "data_stats.csv"
# The first saved prior tables, so they are not hand-picked, and small, well
# known OpenML tables with 3, 2 and 6 classes.
N_PRIOR_EXAMPLES = 3
OPENML_EXAMPLES = ["penguins", "heart-statlog", "breast-tissue"]

# Benchmark names in italic, as in the report text (\tabpfnval).
LEGEND = {
    "prior_val": r"Prior ($\mathit{prior\_val}$)",
    "tabpfn_val": r"OpenML ($\mathit{tabpfn\_val}$)",
}

# Prior in light gray, OpenML in black on top, so the model colors of the
# Section 2 figures keep their meaning.
POINT_STYLES = {
    "prior_val": {"s": 3, "color": INK_MUTED, "alpha": 0.35, "linewidths": 0},
    "tabpfn_val": {"s": 10, "color": INK, "marker": "x", "linewidths": 0.8},
}
BAR_COLORS = {"prior_val": INK_MUTED, "tabpfn_val": INK}


def example_tables() -> tuple[list[TableSample], list[str]]:
    """The tables of Fig. 1 (OpenML split 0) and their panel titles."""
    prior = list(itertools.islice(load_data("prior_val"), N_PRIOR_EXAMPLES))
    datasets = {name: BENCHMARKS["tabpfn_val"][name] for name in OPENML_EXAMPLES}
    openml = OpenMLDataset(datasets, default_data_dir())
    real = [openml.load(name) for name in OPENML_EXAMPLES]
    # Explicit titles: show() would look the prior table number up as an
    # OpenML id and could print a dataset name.
    titles = [f"Prior table {table.dataset_id}" for table in prior]
    return prior + real, titles + OPENML_EXAMPLES


def plot_examples(tables: list[TableSample], titles: list[str], outfile: Path):
    """Prior tables in the top row, OpenML tables below, in the look of the
    other report figures (font, sizes, colors), without the size subtitles
    (the report text gives the sizes that matter)."""
    look: dict[RcKeyType, Any] = {
        **RC_PARAMS,
        "axes.titleweight": "normal",  # Computer Modern has no bold
        "axes.titlelocation": "center",
        "axes.grid": False,  # a grid adds nothing to a projection
    }
    show(
        tables,
        outfile,
        titles=titles,
        ncols=3,
        panel_size=TEXT_WIDTH_IN / 3,
        max_points=250,
        subtitles=False,
        rc=look,
    )
    print(f"Wrote {outfile}")


def plot_shapes(stats: pd.DataFrame, outfile: Path):
    """Shapes of all evaluation tables: training rows vs. features (left) and
    the share of tables per number of classes (right).

    Left: one point per table, log axes; the dashed box is the range the
    prior was trained on (320 to 2,048 training rows, 2 to 100 features), so
    OpenML tables outside it are easy to spot. Uses split 0, so each table
    counts once.
    """
    stats = stats[stats["split_seed"] == 0]
    settings = PRIOR_SETTINGS
    min_rows = round(settings["n_train"] * settings["n_samples"])
    max_rows = round(settings["n_train"] * settings["max_samples"])
    min_features, max_features = settings["min_features"], settings["max_features"]

    use_style()
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_IN, 2.1), width_ratios=[1.6, 1]
    )
    for source, style in POINT_STYLES.items():
        rows = stats[stats["source"] == source]
        left.scatter(rows["n_train"], rows["n_features"], label=LEGEND[source], **style)
    left.add_patch(
        Rectangle(
            (min_rows, min_features),
            max_rows - min_rows,
            max_features - min_features,
            fill=False,
            edgecolor=INK_MUTED,
            linestyle="--",
            linewidth=0.8,
        )
    )
    left.set(xscale="log", yscale="log", xlabel="Training rows", ylabel="Features")
    for axis, ticks in [(left.xaxis, [30, 100, 300, 1000]), (left.yaxis, [2, 10, 100])]:
        axis.set_ticks(ticks, labels=[str(t) for t in ticks])
        axis.set_minor_locator(plt.NullLocator())

    classes = np.arange(2, 11)
    width = 0.4
    for offset, (source, color) in zip(
        (-width / 2, width / 2), BAR_COLORS.items(), strict=True
    ):
        values = stats.loc[stats["source"] == source, "n_classes"]
        # Shares, not counts: 1,600 prior tables vs. 57 OpenML tables.
        share = values.value_counts(normalize=True).reindex(classes, fill_value=0)
        right.bar(classes + offset, share, width, color=color)
    right.set(xlabel="Classes", ylabel="Share of tables", xticks=classes)
    right.grid(axis="x", visible=False)

    handles, labels = left.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
        markerscale=2.5,  # the prior points are tiny in the plot
    )
    fig.tight_layout()
    save(fig, outfile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make the Section 2 figures.")
    parser.add_argument("--stats", type=Path, default=STATS, help="explore.py output")
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    tables, titles = example_tables()
    plot_examples(tables, titles, args.out / "example_tables.pdf")
    plot_shapes(pd.read_csv(args.stats), args.out / "table_shapes.pdf")


if __name__ == "__main__":
    main()
