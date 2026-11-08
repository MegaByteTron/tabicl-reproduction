import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

from tabicl_reproduction.data import TableSample

matplotlib.use("Agg")  # render without a display, e.g. on the cluster


def _make_sample(n_features=2, n_train=60, n_test=20, n_classes=3, dataset_id=None):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n_train + n_test, n_features, generator=g)
    y = torch.arange(n_train + n_test) % n_classes
    return TableSample(
        X[:n_train], y[:n_train], X[n_train:], y[n_train:], dataset_id=dataset_id
    )


@pytest.fixture
def make_sample():
    """Small random table: make_sample(n_features=2, n_classes=3, ...)."""
    return _make_sample


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
