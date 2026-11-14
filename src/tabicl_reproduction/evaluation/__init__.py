"""Scoring models on tables, the same way during pretraining and after.

- metrics.py: the metric registry (@register_metric) and evaluate_metrics.
- models.py: the model registry (@register_model): baselines and ours.
- benchmark.py: score_table, and every model on every table
  (scripts/benchmark.py).
- results.py: the results CSV, saved predictions and the summary.
"""

from .metrics import METRIC_REGISTRY, evaluate_metrics, register_metric
from .models import MODEL_REGISTRY, get_models, register_model

__all__ = [
    "METRIC_REGISTRY",
    "MODEL_REGISTRY",
    "evaluate_metrics",
    "get_models",
    "register_metric",
    "register_model",
]
