import torch

from tabicl_reproduction.data import PriorDataset, synthetic


def test_prior_sample_runs_through_model(check_sample, run_model):
    sample = PriorDataset(n_samples=128, max_features=10).sample()
    check_sample(sample)
    assert (len(sample.X_train), len(sample.X_test)) == (102, 26)  # 80/20
    run_model(sample)


def test_prior_exact_row_counts():
    sample = PriorDataset(n_train=60, n_test=20, max_features=5).sample()
    assert (len(sample.X_train), len(sample.X_test)) == (60, 20)


def test_prior_fraction():
    sample = PriorDataset(n_test=0.25, n_samples=200, max_features=5).sample()
    assert (len(sample.X_train), len(sample.X_test)) == (150, 50)


def test_prior_random_table_size():
    prior = PriorDataset(n_samples=50, max_samples=200, max_features=3)
    sizes = []
    for _ in range(8):
        sample = prior.sample()
        n_rows = len(sample.X_train) + len(sample.X_test)
        assert 50 <= n_rows <= 200
        assert len(sample.X_test) == round(0.2 * n_rows)  # 80/20
        sizes.append(n_rows)
    assert len(set(sizes)) > 1


def test_prior_shuffles_columns(monkeypatch):
    # Column i of the fake prior is filled with the value i, so the values in X
    # show where each column ended up.
    def fake_prior(x_cat_sizes, y_cat_sizes, n_rows):
        columns = {f"x_{i}": torch.full((n_rows, 1), float(i)) for i in range(8)}
        columns["y_0"] = (torch.arange(n_rows) % 2).unsqueeze(-1)
        return columns

    monkeypatch.setattr(synthetic, "randlogint", lambda low, high: 8)
    monkeypatch.setattr(synthetic, "rand_dataset_filtered", fake_prior)
    orders = [PriorDataset(n_samples=20).sample().X_train[0].tolist() for _ in range(3)]
    for order in orders:
        assert sorted(order) == list(range(8))  # every column once
    assert any(order != list(range(8)) for order in orders)
