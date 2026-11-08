import pytest

from tabicl_reproduction.training.optim import (
    MultiOptimizer,
    constant_schedule,
    muon_adamw,
    warmup_cosine,
)

from .conftest import tiny_model


def test_muon_gets_only_matrices():
    model = tiny_model()
    optimizer = muon_adamw(model.parameters())

    muon, adamw = optimizer.optimizers
    assert all(p.ndim == 2 for g in muon.param_groups for p in g["params"])
    assert all(p.ndim != 2 for g in adamw.param_groups for p in g["params"])
    grouped = [p for g in optimizer.param_groups for p in g["params"]]
    assert len(grouped) == len(list(model.parameters()))


def test_scheduler_drives_every_sub_optimizer():
    optimizer = muon_adamw(tiny_model().parameters(), lr=1e-3)
    scheduler = warmup_cosine(optimizer, total_steps=100, warmup_proportion=0.1)

    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[0] == 0.0
    assert lrs[10] == pytest.approx(1e-3)
    assert lrs[-1] < 1e-5
    assert len({g["lr"] for g in optimizer.param_groups}) == 1


def test_multi_optimizer_state_roundtrip():
    optimizer = muon_adamw(tiny_model().parameters())
    fresh = muon_adamw(tiny_model().parameters())
    fresh.load_state_dict(optimizer.state_dict())

    assert isinstance(fresh, MultiOptimizer)
    # Groups stay shared with the sub-optimizers after loading.
    assert fresh.param_groups[0] is fresh.optimizers[0].param_groups[0]


def test_constant_schedule_keeps_lr():
    optimizer = muon_adamw(tiny_model().parameters(), lr=2e-3)
    scheduler = constant_schedule(optimizer, total_steps=10)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 2e-3
