import math

import numpy as np
import pytest
import torch

from tabicl_reproduction.evaluation.metrics import (
    METRIC_REGISTRY,
    evaluate_metrics,
    expected_calibration_error,
    register_metric,
    roc_auc,
)


def test_hand_computed_example():
    y = np.array([0, 0, 0, 1, -1])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
    m = evaluate_metrics(y, proba)
    assert m["accuracy"] == 0.75  # row 4 wrong, row 5 (-1) left out
    assert m["balanced_accuracy"] == 0.5  # class 0: 3/3, class 1: 0/1
    assert m["f1_macro"] == pytest.approx((6 / 7 + 0) / 2)  # class 0 F1 = 6/7
    assert m["log_loss"] == pytest.approx(-np.log([0.9, 0.8, 0.7, 0.4]).mean())
    assert m["roc_auc"] == 1.0  # the class-1 row is ranked above all others
    assert m["ece"] == pytest.approx((0.1 + 0.2 + 0.3 + 0.6) / 4)


def test_every_registered_metric_is_computed():
    m = evaluate_metrics(np.array([0, 1, 2]), np.eye(3))
    assert list(m) == list(METRIC_REGISTRY)
    assert {k: METRIC_REGISTRY[k].higher_is_better for k in ("log_loss", "ece")} == {
        "log_loss": False,
        "ece": False,
    }


def test_multiclass_works():
    # The sklearn scorer version crashed here (roc_auc needs multi_class).
    y = np.array([0, 1, 2, 2])
    proba = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]]
    )
    m = evaluate_metrics(y, proba)
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0


def test_minus_one_rows_are_left_out_of_every_metric():
    y = np.array([0, 1, -1, 1])
    proba = np.array([[0.7, 0.3], [0.4, 0.6], [0.0, 1.0], [0.2, 0.8]])
    assert evaluate_metrics(y, proba) == evaluate_metrics(
        y[[0, 1, 3]], proba[[0, 1, 3]]
    )


def test_only_unseen_classes_raises():
    with pytest.raises(ValueError, match="No test row"):
        evaluate_metrics(np.array([-1, -1]), np.full((2, 2), 0.5))


def test_roc_auc_is_nan_with_one_class():
    m = evaluate_metrics(np.array([1, 1]), np.array([[0.3, 0.7], [0.6, 0.4]]))
    assert math.isnan(m["roc_auc"])
    assert m["accuracy"] == 0.5


def test_roc_auc_handles_binary_and_multiclass():
    proba = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.1, 0.1, 0.8]])
    assert roc_auc(np.array([0, 1, 2]), proba.copy()) == 1.0
    assert roc_auc(np.array([0, 1]), proba[:2].copy()) == 1.0


def test_ece_is_zero_when_confidence_matches_accuracy():
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(np.array([0, 1]), proba) == 0.0
    assert expected_calibration_error(np.array([1, 0]), proba) == 1.0


def test_zero_probability_on_present_classes():
    # Test rows hold classes 0 and 1 only; the last row puts all mass on 2,
    # as k-NN or a random forest can. Nothing may crash or become infinite.
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.6, 0.4, 0], [0.2, 0.8, 0], [0.8, 0.2, 0], [0, 0, 1.0]])
    assert all(math.isfinite(v) for v in evaluate_metrics(y, proba).values())


def test_metrics_do_not_change_the_probabilities():
    proba = np.array([[0.6, 0.4, 0], [0, 0, 1.0]])
    before = proba.copy()
    evaluate_metrics(np.array([0, 1]), proba)
    assert np.array_equal(proba, before)


def test_float_and_tensor_labels_work():
    proba = np.array([[0.7, 0.3], [0.4, 0.6]])
    assert evaluate_metrics(torch.tensor([0.0, 1.0]), proba)["accuracy"] == 1.0


def test_register_a_new_metric():
    @register_metric("test_top_proba")
    def top_proba(y_true, proba):
        return float(proba.max(axis=1).mean())

    try:
        assert (
            evaluate_metrics(np.array([0]), np.array([[0.9, 0.1]]))["test_top_proba"]
            == 0.9
        )
        with pytest.raises(ValueError, match="already registered"):
            register_metric("test_top_proba")(top_proba)
    finally:
        del METRIC_REGISTRY["test_top_proba"]
