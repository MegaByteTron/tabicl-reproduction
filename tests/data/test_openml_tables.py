import os

import openml
import pandas as pd
import pytest
import torch

from tabicl_reproduction.data import OpenMLDataset


def test_openml_from_cache(cached_dataset, check_sample, run_model):
    ds = OpenMLDataset(dataset_ids=(1,), data_dir=cached_dataset, download=False)
    sample = next(iter(ds))
    check_sample(sample)
    assert (len(sample.X_train), len(sample.X_test)) == (32, 8)
    run_model(sample)


def test_openml_split_is_reproducible(cached_dataset):
    ds = OpenMLDataset(dataset_ids=(1,), data_dir=cached_dataset, download=False)
    assert torch.equal(ds.load(1).X_train, ds.load(1).X_train)


def test_openml_respects_max_train(cached_dataset):
    ds = OpenMLDataset(
        dataset_ids=(1,), data_dir=cached_dataset, download=False, max_train=10
    )
    sample = ds.load(1)
    assert (len(sample.X_train), len(sample.X_test)) == (10, 8)


def test_openml_missing_cache_raises(tmp_path):
    ds = OpenMLDataset(dataset_ids=(1,), data_dir=tmp_path, download=False)
    with pytest.raises(FileNotFoundError):
        next(iter(ds))


class FakeOpenMLDataset:
    default_target_attribute = "label"

    def get_data(self, target, dataset_format):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"]})
        y = pd.Series(["p", "q", "p", "q"], name=target)
        return X, y, None, None


@pytest.fixture
def fake_openml(monkeypatch):
    calls = []

    def get_dataset(dataset_id):
        calls.append(dataset_id)
        return FakeOpenMLDataset()

    monkeypatch.setattr(openml.datasets, "get_dataset", get_dataset)
    return calls


@pytest.fixture
def strict_umask():
    """New files are private by default, as on some clusters."""
    old = os.umask(0o077)
    yield
    os.umask(old)


def test_download_makes_everything_readable(tmp_path, fake_openml, strict_umask):
    data_dir = tmp_path / "tabpfn_val"
    OpenMLDataset(dataset_ids=(5, 6), data_dir=data_dir).download_all()

    paths = [data_dir, *data_dir.rglob("*")]
    assert len(paths) == 7  # root, 2 folders, 4 files
    for path in paths:
        mode = path.stat().st_mode
        assert mode & 0o444 == 0o444, f"{path} not readable by all"
        if path.is_dir():
            assert mode & 0o111 == 0o111, f"{path} cannot be opened by all"
    assert not list(data_dir.rglob("*.tmp"))


def test_download_skips_cached_datasets(tmp_path, fake_openml):
    ds = OpenMLDataset(dataset_ids=(5,), data_dir=tmp_path)
    ds.download_all()
    ds.download_all()
    ds.load(5)
    assert fake_openml == [5]  # downloaded only once


def test_download_reports_multi_target_dataset(tmp_path, monkeypatch):
    class MultiTarget(FakeOpenMLDataset):
        name = "birds"
        default_target_attribute = "a,b"

    def get_dataset(dataset_id):
        return MultiTarget() if dataset_id == 7 else FakeOpenMLDataset()

    monkeypatch.setattr(openml.datasets, "get_dataset", get_dataset)
    failures = OpenMLDataset(dataset_ids=(5, 7), data_dir=tmp_path).download_all()

    assert list(failures) == [7]
    assert "no single target" in str(failures[7])
    assert (tmp_path / "5" / "features.parquet").exists()  # others still stored
