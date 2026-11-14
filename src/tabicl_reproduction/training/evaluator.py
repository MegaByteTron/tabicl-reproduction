"""Validation during pretraining, for W&B and best.pt.

A thin adapter: the model is wrapped in the same TabICLClassifier and scored
by the same score_tables as in the final experiment (evaluation/), with one
column order to stay fast. This file only turns the rows into the summary
the trainer logs.
"""

import time
from typing import Any

import numpy as np
import torch
from torch import nn

from tabicl_reproduction.data import load_data
from tabicl_reproduction.evaluation.core import score_tables
from tabicl_reproduction.evaluation.metrics import (  # noqa: F401  (re-exported)
    METRICS as SUMMARY_METRICS,
)
from tabicl_reproduction.evaluation.metrics import (  # noqa: F401  (re-exported)
    expected_calibration_error,
    roc_auc,
)
from tabicl_reproduction.model.classifier import TabICLClassifier


class Evaluator:
    """Scores a model on real tables (default: the tabpfn_val benchmark)."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,  # unused: the classifier uses the model's device
        # Any, not Iterable: the pydantic parser used by the CLI would turn an
        # Iterable into a one-shot iterator, and the second evaluation would
        # find no tables.
        tables: Any = None,
        max_classes: int = 10,
        n_orders: int = 1,
    ):
        self.model = model
        self.device = device
        self.tables = load_data("tabpfn_val") if tables is None else tables
        self.max_classes = max_classes
        self.n_orders = n_orders

    def evaluate(self) -> dict[str, Any]:
        """Per-dataset scores and their means.

        Returns {"summary": {metric: mean, ..., n_datasets, time_s},
        "datasets": [one dict per table]}. Tables the model can't score
        (e.g. more classes than it supports) are left out.
        """
        clf = TabICLClassifier(
            model=self.model, max_classes=self.max_classes, n_orders=self.n_orders
        )
        start = time.perf_counter()
        rows = [
            {k: v for k, v in row.items() if k not in ("status", "error")}
            for row in score_tables(clf, self.tables)
            if row["status"] == "ok"
        ]
        if not rows:
            raise ValueError("No table could be evaluated")
        summary = {m: float(np.nanmean([r[m] for r in rows])) for m in SUMMARY_METRICS}
        summary |= {"n_datasets": len(rows), "time_s": time.perf_counter() - start}
        return {"summary": summary, "datasets": rows}
