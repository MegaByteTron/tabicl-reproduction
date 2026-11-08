from tabicl_reproduction.data import PriorDataset


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
