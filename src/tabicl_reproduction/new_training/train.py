import random

import numpy as np
import schedulefree
import torch as tr
from torch import nn

from tabicl_reproduction.data import load_data
from tabicl_reproduction.model import NanoTabICLv2
from tabicl_reproduction.new_training.trainer import Trainer


def set_seed(seed: int) -> None:
    # The prior draws from the global numpy and torch generators.
    random.seed(seed)
    np.random.seed(seed)
    tr.manual_seed(seed)


def train():
    set_seed(0)

    model = NanoTabICLv2(max_classes=10, out_dim=10)

    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=4e-3, weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion)

    dataloader = load_data()
    trainer.fit(dataloader, n_epochs=1)


def main() -> None:
    train()


if __name__ == "__main__":
    main()
