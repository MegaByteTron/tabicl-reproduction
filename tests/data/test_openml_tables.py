import os

import openml
import pandas as pd
import pytest
import torch

from tabicl_reproduction.data import OpenMLDataset
from tabicl_reproduction.data.benchmarks import TARGET_OVERRIDES

FAKE = {"fake_dataset": 1}  # the dataset written by the cached_dataset fixture


def test_openml_from_cache(cached_dataset, check_sample, run_model):
    ds = OpenMLDataset(datasets=FAKE, data_dir=cached_dataset, download=False)
    sample = next(iter(ds))
    check_sample(sample)
    assert (len(sample.X_train), len(sample.X_test)) == (32, 8)
    run_model(sample)


def test_openml_split_is_reproducible(cached_dataset):
    ds = OpenMLDataset(datasets=FAKE, data_dir=cached_dataset, download=False)
    assert torch.equal(ds.load("fake_dataset").X_train, ds.load("fake_dataset").X_train)


def test_openml_drops_rows_without_a_label(cached_dataset, check_sample):
    target_path = cached_dataset / "fake_dataset" / "target.parquet"
    target = pd.read_parquet(target_path)
    target.loc[[0, 5, 9], "target"] = None  # 3 of the 40 rows are unlabeled
    target.to_parquet(target_path)

    ds = OpenMLDataset(datasets=FAKE, data_dir=cached_dataset, download=False)
    sample = ds.load("fake_dataset")
    check_sample(sample)
    assert len(sample.X_train) + len(sample.X_test) <= 37


def test_openml_respects_max_train(cached_dataset):
    ds = OpenMLDataset(
        datasets=FAKE, data_dir=cached_dataset, download=False, max_train=10
    )
    sample = ds.load("fake_dataset")
    assert (len(sample.X_train), len(sample.X_test)) == (10, 8)


def test_openml_missing_cache_raises(tmp_path):
    ds = OpenMLDataset(datasets=FAKE, data_dir=tmp_path, download=False)
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
    OpenMLDataset(datasets={"five": 5, "six": 6}, data_dir=data_dir).download_all()

    paths = [data_dir, *data_dir.rglob("*")]
    assert len(paths) == 7  # root, 2 folders, 4 files
    for path in paths:
        mode = path.stat().st_mode
        assert mode & 0o444 == 0o444, f"{path} not readable by all"
        if path.is_dir():
            assert mode & 0o111 == 0o111, f"{path} cannot be opened by all"
    assert not list(data_dir.rglob("*.tmp"))


def test_download_skips_cached_datasets(tmp_path, fake_openml):
    ds = OpenMLDataset(datasets={"five": 5}, data_dir=tmp_path)
    ds.download_all()
    ds.download_all()
    ds.load("five")
    assert fake_openml == [5]  # downloaded only once


def test_download_reports_multi_target_dataset(tmp_path, monkeypatch):
    class MultiTarget(FakeOpenMLDataset):
        name = "birds"
        default_target_attribute = "a,b"

    def get_dataset(dataset_id):
        return MultiTarget() if dataset_id == 7 else FakeOpenMLDataset()

    monkeypatch.setattr(openml.datasets, "get_dataset", get_dataset)
    failures = OpenMLDataset(
        datasets={"five": 5, "seven": 7}, data_dir=tmp_path
    ).download_all()

    assert list(failures) == ["seven"]
    assert "no single target" in str(failures["seven"])
    assert (tmp_path / "five" / "features.parquet").exists()  # others still stored


def test_download_uses_target_override(tmp_path, monkeypatch):
    class MultiLabel(FakeOpenMLDataset):
        name = "birds"
        default_target_attribute = "a,b"

        def get_data(self, target, dataset_format):
            X = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 1, 0, 0], "c": [0.5] * 4})
            return X.drop(columns=target), X[target], None, None

    monkeypatch.setattr(openml.datasets, "get_dataset", lambda _: MultiLabel())
    monkeypatch.setitem(TARGET_OVERRIDES, 7, "a")
    failures = OpenMLDataset(datasets={"seven": 7}, data_dir=tmp_path).download_all()

    assert failures == {}
    features = pd.read_parquet(tmp_path / "seven" / "features.parquet")
    target = pd.read_parquet(tmp_path / "seven" / "target.parquet")["target"]
    assert list(features.columns) == ["c"]  # the other label "b" is not a feature
    assert target.tolist() == [0, 1, 0, 1]
