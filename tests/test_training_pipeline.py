import numpy as np
import pytest
import torch

from tabicl_reproduction.classifier import Classifier
from tabicl_reproduction.data import TableSample
from tabicl_reproduction.evaluation import probability_metrics, summarize
from tabicl_reproduction.experiment import (
    Model,
    ModelConfig,
    atomic_save,
    load_checkpoint,
    restore_rng,
    rng_state,
)
from tabicl_reproduction.training import task_loss, training_step


@pytest.fixture
def small():
    torch.set_num_threads(2)
    return Model(
        ModelConfig(max_classes=2, embed_dim=16, n_cls_rows=4, icl_num_blocks=1)
    )


@pytest.fixture
def task():
    X = torch.randn(24, 3)
    y = (X[:, 0] > 0).long()
    return TableSample(X[:16], y[:16], X[16:], y[16:])


def test_training_changes_weights_and_can_fit_one_task(small, task):
    optimizer = torch.optim.AdamW(small.parameters(), lr=0.003)
    initial = task_loss(small, task, "cpu").item()
    for _ in range(30):
        training_step(small, optimizer, [task], "cpu")
    final = task_loss(small, task, "cpu").item()
    assert final < initial * 0.5  # integration diagnostic, NOT a generalization claim


def test_fit_preserves_weights_and_decodes_labels(small, task):
    before = {key: value.clone() for key, value in small.state_dict().items()}
    labels = np.where(task.y_train.numpy() == 0, "no", "yes")
    classifier = Classifier(small, batch_size=2).fit(task.X_train, labels)
    probs = classifier.predict_proba(task.X_test)
    assert probs.shape == (8, 2)
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)
    assert set(classifier.predict(task.X_test)) <= {"no", "yes"}
    for key, value in small.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert small.training  # prediction restores caller's mode


def test_test_chunking_does_not_change_predictions(small, task):
    whole = Classifier(small, batch_size=100).fit(task.X_train, task.y_train)
    chunked = Classifier(small, batch_size=1).fit(task.X_train, task.y_train)
    np.testing.assert_allclose(
        whole.predict_proba(task.X_test), chunked.predict_proba(task.X_test), atol=1e-6
    )


def test_imputation_uses_training_data_only(small):
    X = np.array([[1.0, np.nan], [3.0, np.nan]])
    classifier = Classifier(small).fit(X, [0, 1])
    np.testing.assert_allclose(classifier.imputation_.numpy(), [2, 0])
    classifier.predict_proba([[np.nan, 1000.0]])
    np.testing.assert_allclose(classifier.imputation_.numpy(), [2, 0])


def test_unknown_test_labels_are_counted_as_wrong():
    result = probability_metrics(
        np.array([0, 1, -1]),
        np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]]),
        np.array([0, 1]),
    )
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["known_fraction"] == pytest.approx(2 / 3)
    assert result["balanced_accuracy_known"] == 1
    assert result["log_loss_known"] == pytest.approx(-np.log([0.9, 0.8]).mean())


def test_checkpoint_roundtrip_and_rng(tmp_path, small, task):
    path = tmp_path / "checkpoint.pt"
    state = rng_state()
    expected_random = torch.rand(4)
    atomic_save(
        {
            "format_version": 1,
            "model_config": vars(small.config),
            "model_state": small.state_dict(),
            "rng": state,
        },
        path,
    )
    restored, payload = load_checkpoint(path)
    restore_rng(payload["rng"])
    torch.testing.assert_close(torch.rand(4), expected_random)
    torch.testing.assert_close(
        task_loss(restored, task, "cpu"), task_loss(small, task, "cpu")
    )


def test_summary_compares_only_common_successful_tasks():
    records = [
        {"task": 0, "method": "a", "status": "ok", "accuracy": 0.8},
        {"task": 1, "method": "a", "status": "ok", "accuracy": 0.2},
        {"task": 0, "method": "b", "status": "ok", "accuracy": 0.6},
        {"task": 1, "method": "b", "status": "error"},
    ]
    result = summarize(records)
    assert result["a"]["accuracy"] == 0.5
    assert result["a"]["common_accuracy"] == 0.8
    assert result["b"]["n_failed"] == 1


def test_no_known_test_labels_fails_explicitly(small, task):
    sample = TableSample(
        task.X_train, task.y_train, task.X_test, torch.full_like(task.y_test, -1)
    )
    with pytest.raises(ValueError, match="no test labels"):
        task_loss(small, sample, "cpu")


def test_validation_before_training_keeps_rope_cache_differentiable(small, task):
    Classifier(small).fit(task.X_train, task.y_train).predict_proba(task.X_test)
    optimizer = torch.optim.AdamW(small.parameters(), lr=0.001)
    assert np.isfinite(training_step(small, optimizer, [task], "cpu"))


def test_categorical_vocabulary_does_not_use_test_rows():
    import pandas as pd

    from tabicl_reproduction.data.preprocessing import encode_split_categoricals

    full = pd.DataFrame({"category": pd.Categorical(["b", "c", "a"])})
    train, test = encode_split_categoricals(full.iloc[:2], full.iloc[2:])
    assert train[:, 0].tolist() == [0, 1]
    assert torch.isnan(test[0, 0])


def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    import argparse

    from tabicl_reproduction import training

    def sample(_source):
        X = torch.randn(24, 3)
        y = (X[:, 0] > 0).long()
        return TableSample(X[:16], y[:16], X[16:], y[16:])

    monkeypatch.setattr(training, "valid_task", sample)
    monkeypatch.setattr(training, "fixed_tasks", lambda *_args: [sample(None)])

    def arguments(output, steps, resume=None):
        return argparse.Namespace(
            output=output,
            steps=steps,
            tables_per_step=1,
            eval_every=1,
            save_every=1,
            val_tasks=1,
            threads=2,
            n_train=16,
            n_test=8,
            max_features=3,
            max_classes=2,
            device="cpu",
            seed=42,
            embed_dim=16,
            lr=0.001,
            resume=resume,
            real_data_dir=None,
            real_ids=None,
        )

    full, split = tmp_path / "full", tmp_path / "split"
    training.run(arguments(full, 3))
    training.run(arguments(split, 1))
    training.run(arguments(split, 3, split / "latest.pt"))
    model_full, _ = load_checkpoint(full / "latest.pt")
    model_split, _ = load_checkpoint(split / "latest.pt")
    for name, value in model_full.state_dict().items():
        torch.testing.assert_close(
            value, model_split.state_dict()[name], rtol=0, atol=0
        )
