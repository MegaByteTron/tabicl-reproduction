import pytest
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

from tabicl_reproduction.data import TableSample
from tabicl_reproduction.viz import show


def visible_axes(fig):
    return [ax for ax in fig.axes if ax.axison]


def test_single_sample_and_plain_tuple(make_sample):
    sample = make_sample()
    assert len(visible_axes(show(sample))) == 1
    assert len(visible_axes(show(tuple(sample)))) == 1


def test_grid_hides_unused_panels(make_sample):
    fig = show([make_sample() for _ in range(5)], ncols=4)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 8 and len(visible_axes(fig)) == 5


@pytest.mark.parametrize("suffix", [".png", ".pdf"])
def test_saves_outfile(tmp_path, suffix, make_sample):
    out = tmp_path / f"tables{suffix}"
    show([make_sample()], out)
    assert out.stat().st_size > 0


def test_projection_needs_transform(make_sample):
    with pytest.raises(TypeError, match="TSNE has none"):
        show(make_sample(n_features=5), projection=TSNE())


def test_titles(make_sample):
    fig = show(make_sample(dataset_id=61))
    assert fig.axes[0].get_title(loc="left") == "OpenML 61"  # not a benchmark
    fig = show(make_sample(dataset_id=42585))
    assert fig.axes[0].get_title(loc="left") == "penguins"
    fig = show(make_sample(), titles=["mine"])
    assert fig.axes[0].get_title(loc="left") == "mine"


def test_robust_limits_ignore_outlier(make_sample):
    sample = make_sample()
    X_train = sample.X_train.clone()
    X_train[0] = 1000.0  # one extreme outlier
    sample = TableSample(X_train, sample.y_train, sample.X_test, sample.y_test)

    robust = show(sample).axes[0].get_xlim()
    full = show(sample, robust_limits=False).axes[0].get_xlim()
    assert robust[1] < 10 < 1000 < full[1]


def test_unseen_test_labels_are_drawn(make_sample):
    sample = make_sample()
    y_test = sample.y_test.clone()
    y_test[0] = -1
    show(TableSample(sample.X_train, sample.y_train, sample.X_test, y_test))


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one table"):
        show([])
