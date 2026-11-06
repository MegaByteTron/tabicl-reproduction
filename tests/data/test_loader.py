import pickle
from typing import Any

import numpy as np
import pytest
import torch

import tabicl_reproduction.data.loader as loader_module
from tabicl_reproduction.data import BENCHMARKS, TableSample, load_data
from tabicl_reproduction.data.loader import default_data_dir


def first(loader):
    return next(iter(loader))


def test_prior_through_load_data(check_sample):
    sample = first(load_data("prior", n_train=30, n_test=10, max_features=3))
    check_sample(sample)
    assert (len(sample.X_train), len(sample.X_test)) == (30, 10)


def test_transformation_is_applied():
    def double(sample):
        X_train, y_train, X_test, y_test = sample
        return X_train * 2, y_train, X_test * 2, y_test  # plain tuple is fine

    kwargs: dict[str, Any] = dict(n_train=30, n_test=10, max_features=3)
    plain = first(load_data("prior", **kwargs))
    torch.manual_seed(0)
    np.random.seed(0)
    doubled = first(load_data("prior", transformation=double, **kwargs))

    assert isinstance(doubled, TableSample)
    assert torch.allclose(doubled.X_train, plain.X_train * 2)


def test_sample_unpacks_into_four(check_sample):
    sample = first(load_data("prior", n_train=30, n_test=10, max_features=3))
    X_train, y_train, X_test, y_test = sample
    assert X_train is sample.X_train and y_test is sample.y_test
    assert sample.dataset_id is None  # prior tables have no id


@pytest.fixture
def fake_benchmark(cached_dataset, monkeypatch):
    """Register a benchmark whose only dataset is the fake cached one (id 1)."""
    monkeypatch.setitem(BENCHMARKS, "fake", {"fake_dataset": 1})
    return dict(dataset="fake", data_dir=cached_dataset, download=False)


def test_benchmark_samples_carry_dataset_id(fake_benchmark):
    assert first(load_data(**fake_benchmark)).dataset_id == 1


def test_dataset_id_survives_transformation(fake_benchmark):
    def to_plain_tuple(sample):
        return tuple(sample)

    sample = first(load_data(transformation=to_plain_tuple, **fake_benchmark))
    assert isinstance(sample, TableSample)
    assert sample.dataset_id == 1


def test_dataset_id_survives_pickling(fake_benchmark):
    # DataLoader workers send samples between processes via pickle.
    original = first(load_data(**fake_benchmark))
    sample = pickle.loads(pickle.dumps(original))
    assert sample.dataset_id == 1
    assert torch.equal(sample.X_train, original.X_train)


def test_tabpfn_val_name_with_space(cached_dataset):
    # The task writes the name as "tabpfn val"; both spellings work.
    loader = load_data("tabpfn val", data_dir=cached_dataset, download=False)
    assert loader is not None


def test_unknown_dataset_name():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_data("nope")


def test_default_data_dir_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("TABICL_DATASETS_DIR", str(tmp_path))
    assert default_data_dir("tabpfn_val") == tmp_path / "tabpfn_val"


def test_default_data_dir_cluster_and_local(monkeypatch, tmp_path):
    monkeypatch.delenv("TABICL_DATASETS_DIR", raising=False)
    monkeypatch.setattr(loader_module, "CLUSTER_DATASETS_DIR", tmp_path)
    assert default_data_dir("tabpfn_val") == tmp_path / "tabpfn_val"

    monkeypatch.setattr(loader_module, "CLUSTER_DATASETS_DIR", tmp_path / "missing")
    assert (
        default_data_dir("tabpfn_val")
        == loader_module.LOCAL_DATASETS_DIR / "tabpfn_val"
    )
