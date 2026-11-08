from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from tabicl_reproduction.data import TableSample
from tabicl_reproduction.model import NanoTabICLv2


class Trainer:
    def __init__(
        self,
        model: NanoTabICLv2,
        optimizer: Optimizer,
        criterion: nn.CrossEntropyLoss,
        device: torch.device = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        if device is None:
            self.device = torch.device("cpu")

    def train_step(self, batch: TableSample) -> float:
        X_train, y_train, X_test, y_test = batch

        X_train_test = torch.cat([X_train, X_test], dim=-2)

        if X_train_test.dim() == 2:
            X_train_test = X_train_test.unsqueeze(0)
            y_train = y_train.unsqueeze(0)
            y_test = y_test.unsqueeze(0)

        self.optimizer.zero_grad()

        output = self.model(X_train_test, y_train)

        # [B, N, C] -> [B * N, C]
        output = output.reshape(-1, output.shape[-1])

        # [B, N] -> [B * N]
        targets = y_test.reshape(-1).long()

        loss: torch.Tensor = self.criterion(output, targets).mean()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader: Iterable[TableSample]):
        self.model.train()
        self.optimizer.train()

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            total_loss += self.train_step(batch)
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Training dataloader is empty")

        return total_loss / num_batches

    def fit(self, dataloader: Iterable[TableSample], n_epochs: int):
        for epoch in range(n_epochs):
            loss = self.train_epoch(dataloader)

            print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {loss:.4f}")

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
