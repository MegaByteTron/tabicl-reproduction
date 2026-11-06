import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap

from tabicl_reproduction.viz import show, to_numpy
from tabicl_reproduction.viz.view import make_view


def test_views_by_feature_count():
    rng = np.random.default_rng(0)
    y = np.arange(50) % 3
    assert make_view(rng.normal(size=(50, 1)), y, PCA(2)).kind == "1d"
    assert make_view(rng.normal(size=(50, 2)), y, PCA(2)).kind == "raw"
    view = make_view(rng.normal(size=(50, 6)), y, PCA(2))
    assert view.kind == "projected" and view.xlabel.startswith("PC 1 (")
    assert view.transform(np.zeros((3, 6))).shape == (3, 2)


@pytest.mark.parametrize(
    ("projection", "xlabel"),
    [
        (LinearDiscriminantAnalysis(), "LD 1"),
        (Isomap(n_components=2), "Isomap 1"),
        (KernelPCA(2, kernel="rbf"), "KPC 1"),
    ],
)
def test_other_projections(projection, xlabel, make_sample):
    fig = show(make_sample(n_features=5), projection=projection)
    assert fig.axes[0].get_xlabel().startswith(xlabel)


def test_lda_on_two_classes_gives_one_axis(make_sample):
    # LDA has at most n_classes - 1 axes; the second one becomes jitter.
    lda = LinearDiscriminantAnalysis()
    fig = show(make_sample(n_features=5, n_classes=2), projection=lda)
    assert fig.axes[0].get_ylabel() == "(random jitter)"


def test_to_numpy():
    assert isinstance(to_numpy(torch.ones(2, requires_grad=True)), np.ndarray)
    assert isinstance(to_numpy([1, 2]), np.ndarray)
