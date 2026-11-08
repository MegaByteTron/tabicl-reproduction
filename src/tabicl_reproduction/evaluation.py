"""Same-split baselines and per-table metrics, with explicit failure reporting."""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabicl_reproduction.classifier import Classifier
from tabicl_reproduction.data import (
    TABPFN_VAL_DATASETS,
    OpenMLDataset,
    PriorDataset,
    TableSample,
)
from tabicl_reproduction.experiment import fixed_tasks, load_checkpoint


def probability_metrics(y: np.ndarray, p: np.ndarray, classes: np.ndarray) -> dict:
    if p.shape != (len(y), len(classes)) or not np.isfinite(p).all():
        raise ValueError("Invalid prediction probabilities")
    prediction = classes[p.argmax(axis=1)]
    correct = prediction == y
    known = np.isin(y, classes)
    loss = None
    if known.any():
        index = {label: i for i, label in enumerate(classes)}
        cols = np.array([index[label] for label in y[known]])
        loss = float(-np.log(p[known, cols].clip(1e-15, 1.0)).mean())
    # Unknown (-1) test labels count as wrong. Their true distinct class IDs
    # were discarded upstream, so balanced accuracy is explicitly known-only.
    recalls = [correct[y == c].mean() for c in classes if (y == c).any()]
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy_known": float(np.mean(recalls)) if recalls else None,
        "log_loss_known": loss,
        "known_fraction": float(known.mean()),
    }


def evaluate_tasks(
    model,
    tasks: list[TableSample],
    device: str = "cpu",
    baselines: bool = True,
    seed: int = 42,
) -> list[dict]:
    records = []
    for i, sample in enumerate(tasks):
        X_train, y_train, X_test, y_test = (t.cpu().numpy() for t in sample)
        estimators: dict[str, Any] = {"tabicl": Classifier(model, device=device)}
        if baselines:
            estimators |= {
                "majority": DummyClassifier(strategy="prior"),
                "logistic": make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=1000)
                ),
                "random_forest": RandomForestClassifier(
                    n_estimators=100, random_state=seed, n_jobs=1
                ),
            }
        for name, estimator in estimators.items():
            row = {
                "task": i,
                "dataset_id": sample.dataset_id,
                "method": name,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_features": X_train.shape[1],
                "n_classes": len(np.unique(y_train)),
            }
            start = time.perf_counter()
            try:
                estimator.fit(X_train, y_train)
                p = estimator.predict_proba(X_test)
                row |= probability_metrics(y_test, p, estimator.classes_)
                row["status"] = "ok"
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                row |= {"status": "error", "error": str(exc)}
            row["seconds"] = time.perf_counter() - start
            records.append(row)
    return records


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    successful = defaultdict(set)
    for row in records:
        groups[row["method"]].append(row)
        if row["status"] == "ok":
            successful[row["method"]].add(row["task"])
    common = (
        set.intersection(*(successful[name] for name in groups)) if groups else set()
    )
    output = {}
    for name, rows in groups.items():
        ok = [row for row in rows if row["status"] == "ok"]
        paired = [row for row in ok if row["task"] in common]
        result = {
            "n_ok": len(ok),
            "n_failed": len(rows) - len(ok),
            "n_common": len(paired),
        }
        for key in (
            "accuracy",
            "balanced_accuracy_known",
            "log_loss_known",
            "known_fraction",
        ):
            for prefix, selected in (("", ok), ("common_", paired)):
                values = [row[key] for row in selected if row.get(key) is not None]
                result[prefix + key] = float(np.mean(values)) if values else None
        output[name] = result
    return output


def benchmark_tasks(
    data_dir: Path,
    ids: list[int] | None = None,
    n_train: int = 2048,
    n_test: int | None = None,
    seed: int = 42,
    download: bool = False,
) -> tuple[list[TableSample], list[dict]]:
    source = OpenMLDataset(
        dataset_ids=tuple(ids or TABPFN_VAL_DATASETS.values()),
        data_dir=data_dir,
        n_train=None,
        n_test=0.2 if n_test is None else n_test,
        max_train=min(n_train, 2048),
        seed=seed,
        download=download,
    )
    tasks, errors = [], []
    for dataset_id in source.dataset_ids:
        try:
            tasks.append(source.load(dataset_id))
        except Exception as exc:
            errors.append({"dataset_id": dataset_id, "error": str(exc)})
    return tasks, errors


def write_results(records: list[dict], directory: Path, load_errors=()) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in records for key in row})
    if fields:
        with (directory / "per_table.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    result = {"methods": summarize(records), "load_errors": list(load_errors)}
    (directory / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=["prior", "tabpfn_val"], default="prior")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(".cache/datasets/tabpfn_val")
    )
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, payload = load_checkpoint(args.checkpoint, args.device)
    errors = []
    if args.dataset == "prior":
        tasks = fixed_tasks(
            PriorDataset(**payload["prior_config"]), args.tasks, args.seed
        )
    else:
        tasks, errors = benchmark_tasks(
            args.data_dir, args.ids, args.n_train, args.n_test, args.seed, args.download
        )
    rows = evaluate_tasks(model, tasks, args.device)
    print(json.dumps(write_results(rows, args.output, errors), indent=2))
    if not rows or errors or any(row["status"] != "ok" for row in rows):
        raise SystemExit(
            "Evaluation incomplete; inspect summary.json and per_table.csv"
        )


if __name__ == "__main__":
    main()
