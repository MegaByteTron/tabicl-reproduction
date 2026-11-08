"""Optimizers and learning-rate schedules that the configs can choose from."""

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


class MultiOptimizer(Optimizer):
    """Steps several optimizers as one, so one scheduler can drive all of them.

    The param groups are the sub-optimizers' own dicts, so a learning rate set
    here is the one the sub-optimizer uses.
    """

    def __init__(self, optimizers: list[Optimizer]):
        self.optimizers = optimizers
        super().__init__(self._groups(), defaults={})

    def _groups(self) -> list[dict[str, Any]]:
        return [group for opt in self.optimizers for group in opt.param_groups]

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for optimizer, state in zip(
            self.optimizers, state_dict["optimizers"], strict=True
        ):
            optimizer.load_state_dict(state)
        # Loading replaces the sub-optimizers' group dicts; follow them.
        self.param_groups = self._groups()


def muon_adamw(
    params: Iterable[torch.nn.Parameter],
    lr: float = 8e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.999),
) -> MultiOptimizer:
    """Muon for the 2-D weight matrices, AdamW for everything else (TabICLv2).

    The paper's code runs Muon on every parameter (reshaped to 2-D), but
    torch's Muon only takes matrices, so biases, LayerNorm weights and the
    row CLS tokens get AdamW.
    """
    params = [p for p in params if p.requires_grad]
    muon = torch.optim.Muon(
        [p for p in params if p.ndim == 2],
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        adjust_lr_fn="match_rms_adamw",  # lr * 0.2 * sqrt(max(n, m)), as in the paper
    )
    adamw = torch.optim.AdamW(
        [p for p in params if p.ndim != 2],
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    return MultiOptimizer([muon, adamw])


def constant_schedule(optimizer: Optimizer, total_steps: int) -> LRScheduler:
    """Keeps the optimizer's learning rate (e.g. for schedule-free AdamW)."""
    return LambdaLR(optimizer, lambda step: 1.0)


def warmup_cosine(
    optimizer: Optimizer,
    total_steps: int,
    warmup_proportion: float = 0.01,
    lr_end: float = 1e-7,
) -> LRScheduler:
    """Linear warmup, then one cosine decay down to lr_end (TabICLv2 recipe)."""
    warmup = warmup_proportion * total_steps
    lr = optimizer.param_groups[0]["lr"]
    final = lr_end / lr

    def lr_factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = min((step - warmup) / max(1.0, total_steps - warmup), 1.0)
        return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_factor)
