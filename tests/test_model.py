import torch


def test_tensor_operations() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])

    assert x.sum().item() == 6.0


def predict(model, x, y):
    with torch.no_grad():
        return model(x, y)


def test_constant_training_column_is_ignored(model) -> None:
    # Column 0 is 0 in every training row. A test value of 5 would become ~5e8
    # after standardizing; instead the column is 0 for every row, so the
    # predictions are the same as with the training value in the test rows.
    torch.manual_seed(0)
    x = torch.randn(1, 30, 3)
    x[0, :, 0] = 0.0
    y = torch.randint(0, 2, (1, 20))
    x_other = x.clone()
    x_other[0, 20:, 0] = 5.0
    assert torch.allclose(predict(model, x, y), predict(model, x_other, y))


def test_outliers_are_clipped(model) -> None:
    # Test values far outside the training rows are clipped to +-100, so a
    # value of 1e6 and one of 1e9 give the same predictions.
    torch.manual_seed(0)
    x = torch.randn(1, 30, 3)
    y = torch.randint(0, 2, (1, 20))
    x_other = x.clone()
    x[0, 20:, 1], x_other[0, 20:, 1] = 1e6, 1e9
    out = predict(model, x, y)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, predict(model, x_other, y))
