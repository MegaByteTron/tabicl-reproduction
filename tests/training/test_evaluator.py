import math

import numpy as np
import pytest
import torch
from hydra_zen.third_party.pydantic import pydantic_parser

from tabicl_reproduction.data import TableSample
from tabicl_reproduction.training import Evaluator
from tabicl_reproduction.training.evaluator import (
    SUMMARY_METRICS,
    expected_calibration_error,
    roc_auc,
)

from .conftest import tiny_model


def test_evaluate_returns_summary_and_rows(make_sample):
    model = tiny_model().train()
    result = Evaluator(model, "cpu", tables=[make_sample(), make_sample()]).evaluate()

    for metric in SUMMARY_METRICS:
        assert metric in result["summary"]
    assert result["summary"]["n_datasets"] == 2
    row = result["datasets"][0]
    assert row["dataset_id"] == 7 and row["n_classes"] == 3
    assert row["n_test"] == 7  # the -1 row is left out
    assert 0 <= row["accuracy"] <= 1 and row["log_loss"] >= 0
    assert model.training  # mode restored


class Tables:
    """Re-iterable, like the DataLoader the config builds."""

    def __iter__(self):
        return iter([make_table()])


def make_table():
    y_test = torch.arange(8) % 3
    return TableSample(
        torch.randn(24, 3), torch.arange(24) % 3, torch.randn(8, 3), y_test
    )


def test_evaluates_twice_under_the_cli_parser():
    # The CLI instantiates every config through this parser.
    evaluator = pydantic_parser(Evaluator)(tiny_model(), "cpu", tables=Tables())
    assert evaluator.evaluate()["summary"]["n_datasets"] == 1
    assert evaluator.evaluate()["summary"]["n_datasets"] == 1


def test_skips_tables_with_too_many_classes(make_sample):
    evaluator = Evaluator(tiny_model(), "cpu", tables=[make_sample()], max_classes=2)
    with pytest.raises(ValueError, match="No table"):
        evaluator.evaluate()


def test_roc_auc_handles_binary_multiclass_and_single_class():
    probs = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.1, 0.1, 0.8]])
    assert roc_auc(probs, np.array([0, 1, 2])) == 1.0
    assert roc_auc(probs[:2], np.array([0, 1])) == 1.0
    assert math.isnan(roc_auc(probs, np.array([1, 1, 1])))


def test_ece_is_zero_when_confidence_matches_accuracy():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, np.array([0, 1])) == 0.0
    assert expected_calibration_error(probs, np.array([1, 0])) == 1.0
