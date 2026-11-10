import warnings
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from tabicl_reproduction.viz import DecisionBoundary, show


def test_decision_boundary_on_2d_table(make_sample):
    overlay = DecisionBoundary(KNeighborsClassifier(5), resolution=20)
    ax = show(make_sample(), overlays=[overlay]).axes[0]
    image = ax.get_images()[0].get_array()
    assert image is not None and image.shape == (20, 20, 4)
    # The shading covers exactly the visible area.
    assert tuple(ax.get_images()[0].get_extent()[:2]) == ax.get_xlim()


def test_decision_boundary_single_class_table(make_sample):
    sample = make_sample(n_classes=1)
    overlay = DecisionBoundary(KNeighborsClassifier(5), resolution=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # e.g. division by zero
        ax = show(sample, overlays=[overlay]).axes[0]
    image = ax.get_images()[0].get_array()
    assert image is not None
    alpha = image[..., 3]
    assert np.allclose(alpha, overlay.max_alpha)


def test_decision_boundary_rejects_wide_table(make_sample):
    overlay = DecisionBoundary(KNeighborsClassifier(5))
    with pytest.raises(ValueError, match="exactly 2 features"):
        show(make_sample(n_features=3), overlays=[overlay])


@pytest.mark.parametrize(
    ("classifier", "error"),
    [
        (SVC(), "no predict_proba"),
        (LinearRegression(), "needs a scikit-learn classifier"),
        (torch.nn.Linear(2, 2), "needs a scikit-learn classifier"),
    ],
)
def test_decision_boundary_rejects_non_classifier(classifier, error):
    with pytest.raises(TypeError, match=error):
        DecisionBoundary(classifier)


def test_decision_boundary_accepts_svc_with_probabilities():
    DecisionBoundary(SVC(probability=True))


def test_custom_overlay_gets_view(make_sample):
    seen = []
    show(make_sample(n_features=4), overlays=[lambda ax, s, v: seen.append(v.kind)])
    assert seen == ["projected"]


def test_overlay_without_base_class(make_sample):
    # Any object with a matching __call__ is an overlay; called once per panel.
    @dataclass
    class CountCalls:
        calls: int = 0

        def __call__(self, ax, sample, view):
            self.calls += 1
            assert view.limits is not None  # limits are set before overlays
            ax.axvline(0, zorder=5)

    overlay = CountCalls()
    fig = show([make_sample(), make_sample(n_features=4)], overlays=[overlay])
    assert overlay.calls == 2
    assert len(fig.axes[0].lines) == 1
