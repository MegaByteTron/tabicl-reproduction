import numpy as np
import pytest
import torch

from tabicl_reproduction.model.model import NanoTabICLv2


@pytest.fixture(autouse=True)
def seed():
    """Make every test reproducible (the prior uses global RNGs)."""
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="session")
def model():
    """A small classification model, fast enough for tests."""
    return NanoTabICLv2(
        max_classes=10,
        out_dim=10,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
    ).eval()
