"""Scoring models on tables, the same way during pretraining and after.

- metrics.py: compute_metrics, the only implementation of the metrics.
- core.py: score_table / score_tables, the only place a table gets scored.
- results.py: the results CSV, saved predictions and the summary.
- baselines.py: the scikit-learn baselines.
- experiment.py: every model on every table (scripts/compare.py).
"""

from .core import score_table, score_tables
from .metrics import METRICS, compute_metrics

__all__ = ["METRICS", "compute_metrics", "score_table", "score_tables"]
