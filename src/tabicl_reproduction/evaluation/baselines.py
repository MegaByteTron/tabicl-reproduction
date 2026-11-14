"""Simple baselines, fitted from scratch on every table.

Our model must beat these on prior tables (Milestone 1). Every baseline uses
scikit-learn defaults without tuning; k-NN and logistic regression get
standardized features, because they depend on feature scale. They are scored
by evaluation/experiment.py, exactly like our model.
"""

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_baselines() -> dict[str, Any]:
    """A new, unfitted model of each baseline, by name."""
    return {
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(random_state=0)
        ),
        "random_forest": RandomForestClassifier(random_state=0),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=0),
    }
