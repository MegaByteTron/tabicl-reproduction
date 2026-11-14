import pytest
from sklearn.base import is_classifier
from sklearn.neighbors import KNeighborsClassifier

from tabicl_reproduction.evaluation import models
from tabicl_reproduction.evaluation.models import MODEL_REGISTRY, get_models


def test_baselines_are_fresh_classifiers():
    first = dict(get_models())
    assert sorted(first) == [
        "hist_gradient_boosting",
        "knn",
        "logistic_regression",
        "random_forest",
    ]
    assert all(is_classifier(m) for m in first.values())
    second = dict(get_models())
    assert all(first[name] is not second[name] for name in first)  # never shared


def test_our_model_needs_a_checkpoint(monkeypatch):
    with pytest.raises(ValueError, match="needs a checkpoint"):
        list(get_models(["nano_tabicl_v2"]))

    calls = []
    ours = KNeighborsClassifier()  # stands in for the loaded TabICLClassifier

    def fake_factory(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return ours

    assert MODEL_REGISTRY["nano_tabicl_v2"].needs_checkpoint
    monkeypatch.setitem(
        MODEL_REGISTRY, "nano_tabicl_v2", models.ModelSpec(fake_factory, True)
    )
    got = dict(get_models(["nano_tabicl_v2"], checkpoint="run", n_orders=2))
    assert got == {"nano_tabicl_v2": ours}
    assert calls == [("run", {"n_orders": 2})]


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown models"):
        list(get_models(["xgboost"]))
