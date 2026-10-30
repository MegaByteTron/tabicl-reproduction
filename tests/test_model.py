import torch


def test_tensor_operations() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])

    assert x.sum().item() == 6.0
