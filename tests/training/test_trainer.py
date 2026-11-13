import math

import pytest
import schedulefree
import torch
from torch import nn

from tabicl_reproduction.training import Trainer
from tabicl_reproduction.training.optim import constant_schedule

from .conftest import tiny_model


def test_update_every_accum_steps(make_trainer, make_sample):
    trainer = make_trainer(accum_steps=2)
    before = [p.clone() for p in trainer.model.parameters()]

    assert trainer.train_step(make_sample()) is not None
    assert trainer.step == 0  # still accumulating
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, trainer.model.parameters(), strict=True)
    )

    trainer.train_step(make_sample())
    assert trainer.step == 1
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, trainer.model.parameters(), strict=True)
    )


def test_accumulation_averages_gradients(make_trainer, make_sample):
    samples = [make_sample(), make_sample(n_features=2)]
    trainer = make_trainer(accum_steps=2, clip_norm=None)
    reference = make_trainer(accum_steps=1, clip_norm=None)
    reference.model.load_state_dict(trainer.model.state_dict())

    # Gradient of each table on its own, with the trainer's loss.
    per_table = []
    for sample in samples:
        reference.model.zero_grad()
        X_train, y_train, X_test, _ = (t.unsqueeze(0) for t in sample)
        logits = reference.model(torch.cat([X_train, X_test], dim=1), y_train)[0]
        reference.criterion(logits, sample.y_test).backward()
        per_table.append([p.grad.clone() for p in reference.model.parameters()])

    trainer._update = lambda: None  # keep the gradients for inspection
    for sample in samples:
        trainer.train_step(sample)
    for param, *grads in zip(trainer.model.parameters(), *per_table, strict=True):
        torch.testing.assert_close(param.grad, sum(grads) / 2, rtol=1e-4, atol=1e-6)


def test_clipping_uses_fresh_gradients(make_trainer, make_sample):
    trainer = make_trainer(clip_norm=1e-3)
    trainer.train_step(make_sample())
    # The logged norm is the one before clipping, computed from this step.
    assert trainer.window_metrics()["train/grad_norm"] > 1e-3


def test_table_without_known_test_labels_is_skipped(make_trainer, make_sample):
    trainer = make_trainer()
    sample = make_sample()
    sample.y_test[:] = -1

    assert trainer.train_step(sample) is None
    assert trainer.tables_seen == 1


def test_nonfinite_gradient_skips_the_update(make_trainer, make_sample):
    trainer = make_trainer()
    param = next(trainer.model.parameters())
    param.register_hook(lambda grad: torch.full_like(grad, math.inf))
    before = param.clone()

    trainer.train_step(make_sample())
    assert trainer.skipped_steps == 1 and trainer.step == 1
    torch.testing.assert_close(param, before)  # weights untouched


def test_broken_weights_stop_training(make_trainer, make_sample):
    trainer = make_trainer(max_bad_steps=2)
    with torch.no_grad():
        next(trainer.model.parameters()).fill_(math.nan)  # every loss is NaN now

    trainer.train_step(make_sample())
    assert trainer.skipped_steps == 1
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.train_step(make_sample())


def test_train_epoch_counts_steps(make_trainer, tables):
    trainer = make_trainer(accum_steps=2)
    stats = trainer.train_epoch(tables(), n_steps=3)

    assert trainer.step == 3 and trainer.tables_seen == 6
    assert stats["loss"] > 0 and 0 <= stats["accuracy"] <= 1


def test_stop_request_ends_epoch_after_a_full_step(make_trainer, tables):
    trainer = make_trainer(accum_steps=2)
    trainer.stop_requested = True
    trainer.train_epoch(tables(), n_steps=5)
    assert trainer.step == 0 and trainer._pending == 0

    status = trainer.fit(tables(), n_epochs=1, steps_per_epoch=5)
    assert status == "stopped" and trainer.epoch == 0


def test_stop_logs_a_final_row(make_trainer, tables):
    class Rows:
        def __init__(self):
            self.rows = []

        def log(self, metrics, step):
            self.rows.append(metrics | {"step": step})

    trainer = make_trainer(accum_steps=2)
    trainer.stop_requested, trainer.stop_reason = True, "SIGINT"
    logger = Rows()
    trainer.fit(tables(), n_epochs=1, steps_per_epoch=5, logger=logger)

    last = logger.rows[-1]
    assert last["event"] == "stopped" and last["signal"] == "SIGINT"
    assert last["progress/epoch"] == 0 and last["step"] == trainer.step


def test_inference_switches_schedule_free_weights(make_sample):
    model = tiny_model()
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-2)
    trainer = Trainer(model, optimizer, nn.CrossEntropyLoss(ignore_index=-1), "cpu")
    optimizer.train()
    for _ in range(3):
        trainer.train_step(make_sample())

    train_weights = next(model.parameters()).clone()
    with trainer.inference():
        assert not model.training
        eval_weights = next(model.parameters()).clone()
    assert model.training
    assert not torch.equal(train_weights, eval_weights)
    torch.testing.assert_close(next(model.parameters()), train_weights)


def test_state_dict_roundtrip_restores_everything(make_trainer, tables):
    trainer = make_trainer(accum_steps=1)
    trainer.scheduler = constant_schedule(trainer.optimizer, total_steps=10)
    trainer.train_epoch(tables(), n_steps=2)
    state = trainer.state_dict()
    expected_random = torch.rand(3)

    fresh = make_trainer(accum_steps=1)
    fresh.scheduler = constant_schedule(fresh.optimizer, total_steps=10)
    fresh.load_state_dict(state)

    assert (fresh.step, fresh.tables_seen) == (2, 2)
    for a, b in zip(fresh.model.parameters(), trainer.model.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    # RNG restored: the next random numbers are the same as after saving.
    torch.testing.assert_close(torch.rand(3), expected_random)


def test_window_metrics_keys(make_trainer, tables):
    trainer = make_trainer()
    trainer.train_epoch(tables(), n_steps=2)
    metrics = trainer.window_metrics()

    for key in ["train/loss", "train/lr", "time/data", "time/compute", "prior/n_rows"]:
        assert key in metrics
    assert metrics["progress/tables_seen"] == 2


def test_logged_lr_includes_schedule_free_warmup(make_sample):
    model = tiny_model()
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=1e-2, warmup_steps=100
    )
    trainer = Trainer(model, optimizer, nn.CrossEntropyLoss(ignore_index=-1), "cpu")
    optimizer.train()
    trainer.train_step(make_sample())
    # Early in the warm-up the real lr is far below the configured 1e-2.
    assert trainer.window_metrics()["train/lr"] < 1e-3
