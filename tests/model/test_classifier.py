import numpy as np
import pytest
import torch
from hydra_zen import save_as_yaml
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from tabicl_reproduction.data import PriorDataset
from tabicl_reproduction.model import (
    NanoTabICLv2,
    TabICLClassifier,
    UnsupportedTableError,
)
from tabicl_reproduction.viz import DecisionBoundary, show


def small_table():
    return PriorDataset(n_samples=60, max_features=6, max_classes=4).sample()


def string_labels(n_train=30, n_test=10, n_features=3):
    """Labels like a real dataset: strings, not 0..k-1."""
    rng = np.random.default_rng(0)
    names = np.array(["dog", "cat", "fish"])
    y_train = names[np.arange(n_train) % 3]
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    return X_train, y_train, X_test


def test_one_order_is_a_plain_forward_pass(model):
    table = small_table()
    n_classes = int(table.y_train.max()) + 1
    x = torch.cat([table.X_train, table.X_test]).unsqueeze(0)
    with torch.no_grad():
        logits = model(x, table.y_train.unsqueeze(0))[0, :, :n_classes]
    expected = torch.softmax(logits, dim=-1).numpy()

    clf = TabICLClassifier(model, n_orders=1).fit(table.X_train, table.y_train)
    assert np.allclose(clf.predict_proba(table.X_test), expected, atol=1e-6)


def test_orders_are_averaged(model):
    table = small_table()
    clf = TabICLClassifier(model).fit(table.X_train, table.y_train)
    proba = clf.predict_proba(table.X_test)
    assert proba.shape == (len(table.X_test), int(table.y_train.max()) + 1)
    assert np.allclose(proba.sum(axis=1), 1)
    assert np.array_equal(proba, clf.predict_proba(table.X_test))  # reproducible
    one = TabICLClassifier(model, n_orders=1).fit(table.X_train, table.y_train)
    assert not np.allclose(proba, one.predict_proba(table.X_test))


def test_classes_and_columns_follow_sklearn(model):
    X_train, y_train, X_test = string_labels()
    clf = TabICLClassifier(model).fit(X_train, y_train)
    reference = LogisticRegression().fit(X_train, y_train)
    assert list(clf.classes_) == list(reference.classes_) == ["cat", "dog", "fish"]
    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    pred = clf.predict(X_test)
    assert set(pred) <= {"cat", "dog", "fish"}
    assert list(pred) == list(clf.classes_[proba.argmax(axis=1)])


def test_unsupported_tables_raise(model):
    X = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.raises(UnsupportedTableError, match="needs 2 to 3"):
        TabICLClassifier(model, max_classes=3).fit(X, np.arange(20) % 4)
    with pytest.raises(UnsupportedTableError, match="1 classes"):
        TabICLClassifier(model).fit(X, np.zeros(20))


def test_fit_only_stores_rows(model):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    X_train, y_train, X_test = string_labels()
    TabICLClassifier(model).fit(X_train, y_train).predict_proba(X_test)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_training_mode_is_restored(model):
    X_train, y_train, X_test = string_labels()
    model.train()
    try:
        TabICLClassifier(model).fit(X_train, y_train).predict_proba(X_test)
        assert model.training  # e.g. validation in the middle of pretraining
    finally:
        model.eval()


def test_missing_values_are_filled(model):
    X_train, y_train, X_test = string_labels()
    X_train[0, 0] = X_test[1, 2] = np.nan
    proba = TabICLClassifier(model).fit(X_train, y_train).predict_proba(X_test)
    assert np.isfinite(proba).all()


def test_wrong_number_of_features_raises(model):
    X_train, y_train, X_test = string_labels()
    clf = TabICLClassifier(model).fit(X_train, y_train)
    with pytest.raises(ValueError, match="features"):
        clf.predict_proba(X_test[:, :2])


def test_works_with_clone_and_decision_boundary(model, tmp_path):
    # show() fits a fresh copy (clone) of the classifier on every table.
    assert clone(TabICLClassifier(model)).n_orders == 8
    table = PriorDataset(n_samples=40, max_features=2, max_classes=3).sample()
    overlay = DecisionBoundary(TabICLClassifier(model, n_orders=1))
    show([table], outfile=tmp_path / "boundary.png", overlays=[overlay])
    assert (tmp_path / "boundary.png").exists()


def test_from_checkpoint(tmp_path):
    # A fake training run: the resolved config and best.pt, as training writes them.
    run_dir = tmp_path / "run"
    (run_dir / ".hydra").mkdir(parents=True)
    sizes = {"embed_dim": 16, "col_num_blocks": 1, "row_num_blocks": 1}
    model = NanoTabICLv2(max_classes=10, out_dim=10, icl_num_blocks=1, **sizes)
    torch.save({"model": model.state_dict(), "step": 7}, run_dir / "best.pt")
    config = {
        "_target_": "tabicl_reproduction.model.model.NanoTabICLv2",
        "max_classes": 10,
        "out_dim": "${.max_classes}",  # as in a real run's config
        "icl_num_blocks": 1,
        **sizes,
    }
    save_as_yaml({"model": config}, run_dir / ".hydra" / "config.yaml")

    clf = TabICLClassifier.from_checkpoint(run_dir, n_orders=2)
    assert clf.checkpoint_step == 7 and clf.max_classes == 10 and clf.n_orders == 2
    assert clf.model is not None and not clf.model.training
    X_train, y_train, X_test = string_labels()
    assert clf.fit(X_train, y_train).predict_proba(X_test).shape == (len(X_test), 3)
