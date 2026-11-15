"""Section 2 figures: example tables, and the shapes of all evaluation tables.

    uv run python -m tabicl_reproduction.explore   # writes report/data_stats.csv
    uv run python report/plots/data_overview.py

Writes report/figures/example_tables.pdf (show() on 3 prior and 3 OpenML
tables) and table_shapes.pdf (training rows, features and classes of the
prior tables vs. the OpenML tables).
"""

import argparse
import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.typing import RcKeyType
from style import (
    FIGURES_DIR,
    INK,
    INK_MUTED,
    RC_PARAMS,
    SOURCE_LABELS,
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

# Prior filled in gray, OpenML as an outline, so the model colors of the
# Section 3 figures keep their meaning.
SOURCE_STYLES = {
    "prior_val": {"color": INK_MUTED, "alpha": 0.45},
    "tabpfn_val": {"color": INK, "histtype": "step", "linewidth": 1.0},
}


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
    other report figures (font, sizes, colors). The table sizes go in the
    caption, not in subtitles."""
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


def log_bins(low: float, high: float, n: int, below: int = 0, above: int = 0):
    """n log-spaced bins from low to high (both exact edges), plus `below` and
    `above` bins of the same width outside that range."""
    edges = np.geomspace(low, high, n + 1)
    ratio = edges[1] / edges[0]
    lower = low / ratio ** np.arange(below, 0, -1)
    upper = high * ratio ** np.arange(1, above + 1)
    return np.concatenate([lower, edges, upper])


def plot_shapes(stats: pd.DataFrame, outfile: Path):
    """Share of tables per number of training rows, features and classes.

    Dashed lines: the range the prior was trained on (320 to 2,048 training
    rows, at most 100 features); they are bin edges, so no bar crosses them.
    Uses split 0, so each table counts once.
    """
    stats = stats[stats["split_seed"] == 0]
    settings = PRIOR_SETTINGS
    min_rows = round(settings["n_train"] * settings["n_samples"])
    max_rows = round(settings["n_train"] * settings["max_samples"])
    max_features = settings["max_features"] + 0.5  # between 100 and 101
    panels = [
        (
            "n_train",
            "Training rows",
            log_bins(min_rows, max_rows, 6, below=8),
            [min_rows, max_rows],
        ),
        # Half-integer edges: feature counts are integers.
        (
            "n_features",
            "Features",
            log_bins(1.5, max_features, 10, above=3),
            [max_features],
        ),
        ("n_classes", "Classes", np.arange(1.5, 11), []),
    ]
    ticks = {
        "n_train": [30, 100, 300, 1000],
        "n_features": [2, 10, 30, 100, 300],
        "n_classes": list(range(2, 11)),
    }

    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 1.8), sharey=True)
    for ax, (column, label, bins, limits) in zip(axes, panels, strict=True):
        for source, style in SOURCE_STYLES.items():
            values = stats.loc[stats["source"] == source, column]
            # np.histogram drops values outside the bins without a word.
            outside = values[(values < bins[0]) | (values > bins[-1])]
            if len(outside):
                raise ValueError(f"{column} outside the bins: {list(outside)}")
            # Shares, not counts: 1,600 prior tables vs. 57 OpenML tables.
            weights = np.full(len(values), 1 / len(values))
            ax.hist(values, bins, weights=weights, label=SOURCE_LABELS[source], **style)
        for x in limits:
            ax.axvline(x, color=INK_MUTED, linestyle="--", linewidth=0.8)
        if column != "n_classes":
            ax.set_xscale("log")
            ax.minorticks_off()
        ax.set_xticks(ticks[column], labels=[str(t) for t in ticks[column]])
        ax.set_xlabel(label)
    axes[0].set_ylabel("Share of tables")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
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
