import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def cached_dataset(tmp_path):
    """Write a fake OpenML dataset ("fake_dataset", id 1, 40 rows) to disk.

    Returns the data folder that holds it.
    """
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": pd.Series(rng.choice(["x", "y", None], size=n), dtype="object"),
        }
    )
    X.loc[3, "a"] = np.nan
    y = pd.Series(rng.choice(["cat", "dog", "fish"], size=n))
    folder = tmp_path / "fake_dataset"
    folder.mkdir()
    X.to_parquet(folder / "features.parquet")
    y.to_frame(name="target").to_parquet(folder / "target.parquet")
    return tmp_path


def _check_sample(sample):
    X_train, y_train, X_test, y_test = sample
    assert X_train.dim() == 2 and X_test.dim() == 2
    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert X_train.dtype == torch.float32 and y_train.dtype == torch.long
    assert not X_train.isnan().any() and not X_test.isnan().any()
    # Training labels are consecutive from 0.
    assert y_train.unique().tolist() == list(range(int(y_train.max()) + 1))


@pytest.fixture
def check_sample():
    """Assert that a sample has valid shapes, dtypes, no NaN and clean labels."""
    return _check_sample


@pytest.fixture
def run_model(model):
    """Run a sample through the model and assert the output is valid."""

    def run(sample):
        X_train, y_train, X_test, _ = (t.unsqueeze(0) for t in sample)
        x = torch.cat([X_train, X_test], dim=1)
        with torch.no_grad():
            out = model(x, y_train)
        assert out.shape == (1, len(sample.X_test), 10)
        assert torch.isfinite(out).all()

    return run
