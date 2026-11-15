import numpy as np
import pytest
import torch
from sklearn.base import clone

from tabicl_reproduction.model.classifier import NanoTabICLv2Classifier


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = rng.integers(0, 3, 30)
    return X[:20], y[:20], X[20:]


def test_predict_proba_shape_and_sums(model, data) -> None:
    X_train, y_train, X_test = data
    proba = NanoTabICLv2Classifier(model).fit(X_train, y_train).predict_proba(X_test)
    assert proba.shape == (10, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_predict_returns_class_indices(model, data) -> None:
    X_train, y_train, X_test = data
    clf = NanoTabICLv2Classifier(model).fit(X_train, y_train)
    assert clf.num_classes == 3
    assert set(clf.predict(X_test)) <= {0, 1, 2}


def test_matches_direct_forward_pass(model, data) -> None:
    X_train, y_train, X_test = data
    clf = NanoTabICLv2Classifier(model).fit(X_train, y_train)
    x = torch.tensor(np.concatenate((X_train, X_test)), dtype=torch.float32)[None]
    y = torch.tensor(clf.y_train_, dtype=torch.long)[None]
    with torch.no_grad():
        expected = torch.softmax(model(x, y)[0, :, :3], dim=1).numpy()
    np.testing.assert_allclose(clf.predict_proba(X_test), expected, rtol=1e-5)


def test_weights_and_training_flag_unchanged(model, data) -> None:
    X_train, y_train, X_test = data
    before = [p.clone() for p in model.parameters()]
    model.train()
    try:
        NanoTabICLv2Classifier(model).fit(X_train, y_train).predict_proba(X_test)
        assert model.training
    finally:
        model.eval()
    assert all(
        torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)
    )


def test_device_none_keeps_model_where_it_is(model, data) -> None:
    X_train, y_train, X_test = data
    NanoTabICLv2Classifier(model).fit(X_train, y_train).predict_proba(X_test)
    assert next(model.parameters()).device.type == "cpu"


def test_device_moves_model_in_fit(model, data) -> None:
    X_train, y_train, X_test = data
    clf = NanoTabICLv2Classifier(model, device="cpu").fit(X_train, y_train)
    assert clf.predict_proba(X_test).shape == (10, 3)


def test_clone(model) -> None:
    assert isinstance(clone(NanoTabICLv2Classifier(model)), NanoTabICLv2Classifier)
