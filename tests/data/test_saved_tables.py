import json

import pytest
import torch

from tabicl_reproduction.data import load_data
from tabicl_reproduction.prior_val import PRIOR_SETTINGS, save_prior_tables
from tabicl_reproduction.training.configs import PaperModel, PriorData


def as_lists(tables):
    return [(t.dataset_id, list(t)) for t in tables]


def assert_same(a, b):
    assert [i for i, _ in a] == [i for i, _ in b]
    for (_, x), (_, y) in zip(a, b, strict=True):
        assert all(torch.equal(u, v) for u, v in zip(x, y, strict=True))


def test_save_and_load_back(tmp_path, check_sample):
    save_prior_tables(tmp_path, n_tables=3)
    first = as_lists(load_data("prior_val", data_dir=tmp_path))
    second = as_lists(load_data("prior_val", data_dir=tmp_path))

    assert [i for i, _ in first] == [0, 1, 2]
    assert_same(first, second)  # identical every time
    for _, tensors in first:
        check_sample(tensors)
        assert len(tensors[0]) == 819 and len(tensors[2]) == 205  # 1024, 80/20

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["n_tables"] == 3 and meta["version"] == "v1"


def test_same_seed_gives_same_tables(tmp_path):
    save_prior_tables(tmp_path / "a", n_tables=2)
    save_prior_tables(tmp_path / "b", n_tables=2)
    assert_same(
        as_lists(load_data("prior_val", data_dir=tmp_path / "a")),
        as_lists(load_data("prior_val", data_dir=tmp_path / "b")),
    )


def test_workers_do_not_repeat_tables(tmp_path):
    save_prior_tables(tmp_path, n_tables=3)
    tables = load_data("prior_val", data_dir=tmp_path, num_workers=2)
    assert sorted(t.dataset_id for t in tables) == [0, 1, 2]


def test_refuses_to_overwrite(tmp_path):
    save_prior_tables(tmp_path, n_tables=1)
    with pytest.raises(FileExistsError):
        save_prior_tables(tmp_path, n_tables=1)


def test_missing_tables_and_fixed_splits(tmp_path):
    with pytest.raises(FileNotFoundError, match="tabicl_reproduction.prior_val"):
        next(iter(load_data("prior_val", data_dir=tmp_path)))
    with pytest.raises(ValueError, match="fixed splits"):
        load_data("prior_val", data_dir=tmp_path, n_train=10)


def test_settings_match_pretraining():
    # In training, max_classes comes from the model config.
    training, model = vars(PriorData), vars(PaperModel)
    assert training["max_classes"] == "${model.max_classes}"
    assert PRIOR_SETTINGS["max_classes"] == model["max_classes"]
    for name in PRIOR_SETTINGS.keys() - {"max_classes"}:
        assert training[name] == PRIOR_SETTINGS[name], name


def test_incomplete_set_is_refused(tmp_path):
    save_prior_tables(tmp_path, n_tables=2)
    (tmp_path / "prior_0001.pt").unlink()  # e.g. a deleted or missing file
    with pytest.raises(ValueError, match="has 1 tables, meta.json says 2"):
        list(load_data("prior_val", data_dir=tmp_path))
    (tmp_path / "meta.json").unlink()  # e.g. generation stopped with Ctrl-C
    with pytest.raises(FileNotFoundError, match="No complete set"):
        list(load_data("prior_val", data_dir=tmp_path))
