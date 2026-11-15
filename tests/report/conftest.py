import matplotlib
import numpy as np
import pandas as pd
import pytest

from tabicl_reproduction.results import COLUMNS

matplotlib.use("Agg")  # render without a display

MODELS = ("tabicl_best", "random_forest", "knn")


@pytest.fixture
def results() -> pd.DataFrame:
    """A fake load_results() frame: 3 models, 6 prior and 6 OpenML tables
    (2 splits), table sizes spread over all bins of CATEGORIES."""
    rng = np.random.default_rng(0)
    rows = []
    for source, ids, seeds in [
        ("prior_val", [f"prior_{i:04d}" for i in range(6)], [0]),
        ("tabpfn_val", ["1037", "40707", "459", "465", "1060", "1064"], [0, 1]),
    ]:
        for i, table_id in enumerate(ids):
            for seed in seeds:
                for model in MODELS:
                    rows.append(
                        {
                            "model": model,
                            "source": source,
                            "table_id": table_id,
                            "split_seed": seed,
                            "checkpoint_step": 100 if model == "tabicl_best" else None,
                            "n_train": [100, 1000, 2000][i % 3],
                            "n_test": 200,
                            "n_features": [5, 20, 50][i % 3],
                            "n_classes": [2, 4, 8][i // 2],
                            "accuracy": rng.uniform(0.5, 1),
                            "balanced_accuracy": rng.uniform(0.5, 1),
                            "log_loss": rng.uniform(0.1, 1),
                            "roc_auc": rng.uniform(0.6, 1),
                            "ece": rng.uniform(0, 0.2),
                            "status": "ok",
                        }
                    )
    return pd.DataFrame(rows).reindex(columns=list(COLUMNS))


@pytest.fixture
def run_logs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fake metrics.jsonl and eval_datasets.jsonl of a run, as load_run returns."""
    steps = np.arange(10, 210, 10)
    train = pd.DataFrame({"step": steps, "train/loss": np.linspace(2, 0.5, 20)})
    eval_steps = steps[steps % 50 == 0]
    evals = pd.DataFrame({"step": eval_steps, "eval/log_loss": [0.9, 0.6, 0.5, 0.55]})
    metrics = pd.concat([train, evals], ignore_index=True)
    per_dataset = pd.DataFrame(
        [
            {"step": s, "dataset_id": d, "roc_auc": 0.7 + s / 1000}
            for s in eval_steps
            for d in (1037, 40707, 459)
        ]
    )
    return metrics, per_dataset
