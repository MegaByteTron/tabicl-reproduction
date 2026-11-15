"""LaTeX tables of the results, written by script so the numbers never go stale.

Uses the same per-table means as the figures (comparison.per_table): splits
are averaged first and only tables every model scored are counted.
"""

from pathlib import Path

import pandas as pd
from comparison import LOWER_IS_BETTER, SOURCES, per_table
from style import model_label

COLUMNS = {
    "roc_auc": r"ROC AUC $\uparrow$",
    "accuracy": r"Acc.\ $\uparrow$",
    "balanced_accuracy": r"Bal.\ acc.\ $\uparrow$",
    "log_loss": r"Log loss $\downarrow$",
    "ece": r"ECE $\downarrow$",
}
SOURCE_NAMES = {
    "prior_val": r"Prior tables (\texttt{prior\_val})",
    "tabpfn_val": r"OpenML tables (\tabpfnval{})",
}


def mean_scores(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, int]:
    """Mean of every metric per model (rows) on source, and the table count."""
    means = {metric: per_table(df, source, metric).mean() for metric in COLUMNS}
    return pd.DataFrame(means), len(per_table(df, source))


def results_table(df: pd.DataFrame, outfile: str | Path) -> Path:
    """Write the main results table (booktabs) to outfile, for \\input{}.

    One block per source, one row per model; the best value of each column
    and source is bold.
    """
    lines = [
        r"\begin{tabular}{l" + "r" * len(COLUMNS) + "}",
        r"\toprule",
        " & ".join(["Model", *COLUMNS.values()]) + r" \\",
    ]
    for source in SOURCES:
        means, n_tables = mean_scores(df, source)
        lines += [
            r"\midrule",
            rf"\multicolumn{{{len(COLUMNS) + 1}}}{{l}}"
            rf"{{\emph{{{SOURCE_NAMES[source]}, {n_tables} tables}}}} \\",
        ]
        for model, row in means.iterrows():
            cells = [model_label(str(model))]
            for metric, value in row.items():
                column = means[metric].round(3)
                best = column.min() if metric in LOWER_IS_BETTER else column.max()
                text = f"{value:.3f}"
                cells.append(rf"\textbf{{{text}}}" if round(value, 3) == best else text)
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]

    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")
    return path
