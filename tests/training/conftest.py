import pytest
import torch
from torch import nn

from tabicl_reproduction.data import TableSample
from tabicl_reproduction.model import NanoTabICLv2
from tabicl_reproduction.training import Trainer


def tiny_model() -> NanoTabICLv2:
    return NanoTabICLv2(
        max_classes=10,
        out_dim=10,
        embed_dim=16,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
        col_nhead=2,
        row_nhead=2,
        icl_nhead=2,
    )


@pytest.fixture
def make_sample():
    """Random table with 3 classes; the last test row has the unseen label -1."""

    def make(n_train: int = 24, n_test: int = 8, n_features: int = 3) -> TableSample:
        y_test = torch.arange(n_test) % 3
        y_test[-1] = -1
        return TableSample(
            torch.randn(n_train, n_features),
            torch.arange(n_train) % 3,
            torch.randn(n_test, n_features),
            y_test,
            dataset_id=7,
        )

    return make


@pytest.fixture
def make_trainer():
    def make(**kwargs) -> Trainer:
        model = tiny_model()
        optimizer = kwargs.pop(
            "optimizer", torch.optim.AdamW(model.parameters(), lr=1e-3)
        )
        return Trainer(
            model,
            optimizer,
            nn.CrossEntropyLoss(ignore_index=-1),
            device="cpu",
            **kwargs,
        )

    return make


@pytest.fixture
def tables(make_sample):
    """Endless stream of random tables."""

    def stream():
        while True:
            yield make_sample()

    return stream
