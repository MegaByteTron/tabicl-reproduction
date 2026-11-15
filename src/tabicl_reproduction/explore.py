"""Shape and quality numbers of the tables we evaluate on (report Section 2).

From the repo root:

    uv run python -m tabicl_reproduction.explore --limit 5   # quick test
    uv run python -m tabicl_reproduction.explore

Uses the same tables as the baselines and our model (tabpfn_val with 2 splits,
then the 1,600 saved prior tables). Writes one row per table and split to
report/data_stats.csv and prints how many tables have each problem.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from tabicl_reproduction.baselines import result_tables
from tabicl_reproduction.data import BENCHMARKS, OpenMLDataset, TableSample
from tabicl_reproduction.data.loader import default_data_dir
from tabicl_reproduction.results import make_table_id


def table_stats(table: TableSample) -> dict[str, float]:
    """Numbers of one table as the model sees it (encoded, imputed, split)."""
    X_train, y_train, X_test, y_test = table
    counts = torch.bincount(y_train)
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0, unbiased=False)
    varies = std > 0
    # Test values in training standard deviations, as the model scales them.
    z = ((X_test[:, varies] - mean[varies]) / std[varies]).abs()

    train_pairs = set(zip(map(tuple, X_train.tolist()), y_train.tolist(), strict=True))
    train_rows = {row for row, _ in train_pairs}
    test_rows = list(map(tuple, X_test.tolist()))
    return {
        "n_train": len(X_train),
        # All test rows; the results CSVs count only rows of seen classes.
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "n_classes": len(counts),
        "majority_share": counts.max().item() / len(y_train),
        "unseen_test_rows": int((y_test == -1).sum()),
        "one_value_columns": int((~varies).sum()),
        "duplicate_test_rows": sum(row in train_rows for row in test_rows),
        "duplicate_test_rows_same_label": sum(
            pair in train_pairs for pair in zip(test_rows, y_test.tolist(), strict=True)
        ),
        "max_abs_z": z.max().item() if z.numel() else 0.0,
    }


def raw_stats(X: pd.DataFrame) -> dict[str, float]:
    """Numbers that only the raw OpenML data has (imputation removes the NaNs)."""
    return {
        "missing_share": float(X.isna().to_numpy().mean()),
        # The same rule encode_categoricals uses to find categorical columns.
        "categorical_columns": sum(
            not pd.api.types.is_numeric_dtype(X[column]) for column in X.columns
        ),
    }


def collect(limit: int | None = None) -> pd.DataFrame:
    """One row per table and split. Prior tables get NaN for the raw numbers:
    the prior makes no missing values and doesn't record which columns are
    categorical."""
    datasets = BENCHMARKS["tabpfn_val"]
    names = {str(dataset_id): name for name, dataset_id in datasets.items()}
    openml = OpenMLDataset(datasets, default_data_dir(), download=False)
    rows = []
    for info, tables in result_tables(limit):
        source = info["source"]
        for table in tables:
            row = {
                "source": source,
                "table_id": make_table_id(source, table.dataset_id),
                "split_seed": info["split_seed"],
                **table_stats(table),
            }
            if source == "tabpfn_val":
                # The whole raw dataset, not only the rows sampled for the split.
                X, y = openml._load_dataset(names[row["table_id"]])
                row |= raw_stats(X[y.notna()])  # load drops unlabeled rows too
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per source and split: how many tables have each problem."""
    keys = ["source", "split_seed"]

    def has(column: str, threshold: float = 0) -> pd.Series:
        # NaN (not measured) stays NaN instead of counting as "no".
        return (df[column] > threshold).astype(float).where(df[column].notna())

    flags = pd.DataFrame(
        {
            "tables": 1,
            "duplicates": has("duplicate_test_rows"),
            "duplicates_same_label": has("duplicate_test_rows_same_label"),
            "unseen_class": has("unseen_test_rows"),
            "one_value_column": has("one_value_columns"),
            "majority_over_90%": has("majority_share", 0.9),
            "abs_z_over_10": has("max_abs_z", 10),
            "abs_z_over_100": has("max_abs_z", 100),
            "missing_values": has("missing_share"),
            "categorical_column": has("categorical_columns"),
        }
    )
    summary = pd.concat([df[keys], flags], axis=1).groupby(keys).sum(min_count=1)
    groups = df.groupby(keys)
    # Share of all test rows of the source, so big tables weigh more.
    summary["duplicate_row_share"] = (
        groups["duplicate_test_rows"].sum() / groups["n_test"].sum()
    )
    summary["median_majority_share"] = groups["majority_share"].median()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistics of the result tables.")
    parser.add_argument("--limit", type=int, help="tables per source, quick test")
    parser.add_argument(
        "--out",
        type=Path,
        help="default: report/data_stats.csv (results_test/ with --limit)",
    )
    args = parser.parse_args()
    # Not in results/: every CSV there must be a results file.
    folder = "report" if args.limit is None else "results_test"
    out = args.out or Path(folder) / "data_stats.csv"

    df = collect(args.limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4g")
    print(summarize(df).T.round(3).to_string())
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
