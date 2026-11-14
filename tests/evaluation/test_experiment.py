import pytest
import torch

from tabicl_reproduction.data import TableSample
from tabicl_reproduction.data.prior_val import save_prior_tables
from tabicl_reproduction.evaluation import experiment
from tabicl_reproduction.evaluation.baselines import make_baselines
from tabicl_reproduction.evaluation.experiment import Source, run, sources
from tabicl_reproduction.evaluation.results import load_results
from tabicl_reproduction.model import TabICLClassifier


def make_table(dataset_id, n_classes=3, split=0, n_train=30, n_test=10):
    """Easy table: the class is the rounded first feature, plus noise."""
    y_train = torch.arange(n_train) % n_classes
    y_test = torch.arange(n_test) % n_classes
    X_train = torch.randn(n_train, 4) * 0.1
    X_test = torch.randn(n_test, 4) * 0.1
    X_train[:, 0] += y_train
    X_test[:, 0] += y_test
    return TableSample(X_train, y_train, X_test, y_test, dataset_id, split)


def fake_sources():
    openml = [make_table(31, split=0), make_table(31, split=1), make_table(40, 1)]
    prior = [make_table(0), make_table(1, n_classes=2)]
    return [
        Source("tabpfn_val", openml, len(openml)),
        Source("prior_val", prior, len(prior), data_version="v2"),
    ]


def test_every_model_on_every_table(tmp_path, model):
    estimators = make_baselines() | {"tabicl": TabICLClassifier(model, n_orders=1)}
    run(estimators, tmp_path, fake_sources())

    df = load_results(tmp_path / "*.csv")
    assert sorted(df["model"].unique()) == sorted(estimators)
    for _, rows in df.groupby("model"):
        assert list(rows["table_id"]) == ["31", "31", "40", "prior_0000", "prior_0001"]
        assert list(rows["split"]) == [0, 1, 0, 0, 0]
        assert list(rows["status"]) == ["ok", "ok", "skipped", "ok", "ok"]
    assert set(df[df["status"] == "skipped"]["error"]) == {
        "fewer than 2 classes in training"
    }
    assert set(df[df["source"] == "prior_val"]["data_version"]) == {"v2"}
    # One prediction file per scored table and model.
    files = sorted((tmp_path / "predictions" / "tabicl").rglob("*.npz"))
    names = [f"{f.parent.name}/{f.name}" for f in files]
    assert names == [
        "prior_val/prior_0000_split0.npz",
        "prior_val/prior_0001_split0.npz",
        "tabpfn_val/31_split0.npz",
        "tabpfn_val/31_split1.npz",
    ]


def test_refuses_to_append_to_existing_results(tmp_path, model):
    estimators = {"tabicl": TabICLClassifier(model, n_orders=1)}
    run(estimators, tmp_path, fake_sources(), save_predictions=False)
    assert not (tmp_path / "predictions").exists()
    with pytest.raises(SystemExit, match="already has results for"):
        run(estimators, tmp_path, fake_sources())


def test_command_writes_the_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(experiment, "sources", lambda cv, limit: fake_sources())
    experiment.main(["--out", str(tmp_path), "--models", "knn", "random_forest"])
    assert (tmp_path / "summary.txt").exists()
    assert "random_forest" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="Unknown models"):
        experiment.main(["--out", str(tmp_path / "x"), "--models", "tabicl_best"])


def test_full_run_refuses_a_small_test_set(tmp_path, monkeypatch):
    monkeypatch.setenv("TABICL_DATASETS_DIR", str(tmp_path))
    save_prior_tables(tmp_path / "prior_val", n_tables=2)
    with pytest.raises(SystemExit, match="has 2 tables, the full run needs 1600"):
        sources(cv=None)
