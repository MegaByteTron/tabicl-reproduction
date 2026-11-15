import matplotlib
import pandas as pd
import pytest
from data_overview import plot_examples, plot_shapes

from tabicl_reproduction.data import PriorDataset


@pytest.fixture(autouse=True)
def keep_rcparams():
    # plot_shapes() sets the report style globally; don't let it reach other tests.
    with matplotlib.rc_context():
        yield


def shape_stats(n_train=(40, 320, 2048)):
    rows = [
        {"source": source, "split_seed": seed, "n_train": n, "n_features": f}
        | {"n_classes": c}
        for source, seed in [("prior_val", 0), ("tabpfn_val", 0), ("tabpfn_val", 1)]
        for n, f, c in zip(n_train, (2, 100, 299), (2, 5, 10), strict=True)
    ]
    return pd.DataFrame(rows)


def test_shapes_plot_writes_pdf(tmp_path):
    out = tmp_path / "shapes.pdf"
    plot_shapes(shape_stats(), out)
    assert out.stat().st_size > 0


def test_examples_plot_writes_pdf(tmp_path):
    tables = [PriorDataset(n_samples=60, max_features=5).sample() for _ in range(6)]
    out = tmp_path / "examples.pdf"
    plot_examples(tables, [f"table {i}" for i in range(6)], out)
    assert out.stat().st_size > 0
