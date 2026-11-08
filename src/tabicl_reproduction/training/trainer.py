import contextlib
import math
import random
import time
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from tabicl_reproduction.data import TableSample, to_model_inputs
from tabicl_reproduction.model import NanoTabICLv2


class Logger(Protocol):
    def log(self, metrics: dict[str, Any], step: int) -> None: ...
    def log_datasets(self, rows: list[dict[str, Any]], step: int) -> None: ...


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision: str, device: torch.device) -> torch.dtype | None:
    """Autocast dtype for the forward pass; None means plain float32."""
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "auto":
        # A100/H100/H200 support bf16; older GPUs (e.g. Quadro RTX 6000) don't.
        bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if bf16 else None
    raise ValueError(f"Unknown precision {precision!r}, use auto, bf16 or fp32")


class Trainer:
    """Trains NanoTabICLv2 on a stream of tables.

    Tables differ in their number of features, so they cannot be stacked into
    one batch. Every forward pass sees one table, and the gradients of
    accum_steps tables are averaged into one optimizer update ("step").
    """

    def __init__(
        self,
        model: NanoTabICLv2,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device | str | None = None,
        *,
        scheduler: LRScheduler | None = None,
        accum_steps: int = 1,
        clip_norm: float | None = 1.0,
        precision: str = "auto",
        max_bad_steps: int = 10,
    ):
        self.device = select_device() if device is None else torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.accum_steps = accum_steps
        self.clip_norm = clip_norm
        self.autocast_dtype = resolve_precision(precision, self.device)
        self.max_bad_steps = max_bad_steps

        self.step = 0  # optimizer updates so far
        self.epoch = 0  # finished epochs
        self.tables_seen = 0
        self.skipped_steps = 0  # updates dropped because of a non-finite gradient
        self.train_time = 0.0  # seconds spent training, summed over resumed jobs
        self.stop_requested = False  # set by a signal handler before a time limit
        self.last_accuracy = math.nan  # test accuracy of the latest table

        self._pending = 0  # tables backpropagated since the last update
        self._nonfinite_in_step = 0  # tables of the current update with a NaN/inf loss
        self._bad_in_a_row = 0
        self._window: defaultdict[str, float] = defaultdict(float)
        self._window_start = time.perf_counter()

    # ----- modes -----

    def _set_optimizer_mode(self, train: bool) -> None:
        # Schedule-free optimizers evaluate at averaged weights and need to
        # be told which mode they are in; other optimizers have no such method.
        method = getattr(self.optimizer, "train" if train else "eval", None)
        if callable(method):
            method()

    @contextlib.contextmanager
    def inference(self) -> Generator[None]:
        """Model and optimizer in evaluation mode (the weights to evaluate and ship)."""
        was_training = self.model.training
        self.model.eval()
        self._set_optimizer_mode(train=False)
        try:
            yield
        finally:
            if was_training:
                self.model.train()
                self._set_optimizer_mode(train=True)

    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.autocast_dtype)

    # ----- training -----

    def _record(self, **values: float) -> None:
        for name, value in values.items():
            self._window[name] += value

    def train_step(self, batch: TableSample) -> float | None:
        """Forward and backward pass for one table.

        The weights are updated after every accum_steps tables. Returns the
        loss, or None if the table had nothing to learn from.
        """
        start = time.perf_counter()
        x, y = to_model_inputs(batch)
        target = batch.y_test.to(self.device)
        valid = target >= 0  # -1 marks a test class that never occurs in training
        loss_value = None

        if valid.any():
            with self._autocast():
                logits = self.model(x.to(self.device), y.to(self.device))[0]
            loss = self.criterion(logits.float(), target)
            if torch.isfinite(loss):
                (loss / self.accum_steps).backward()
                loss_value = loss.item()
                correct = logits.argmax(dim=-1)[valid] == target[valid]
                self.last_accuracy = correct.float().mean().item()
                self._record(loss=loss_value, accuracy=self.last_accuracy, losses=1)
            else:
                self._nonfinite_in_step += 1
                self._record(nonfinite_losses=1)

        self.tables_seen += 1
        self._pending += 1
        self._record(
            tables=1,
            n_features=x.shape[-1],
            n_rows=x.shape[-2],
            n_classes=int(batch.y_train.max()) + 1,
        )
        if self._pending == self.accum_steps:
            self._update()
        self._record(compute_time=time.perf_counter() - start)
        return loss_value

    def _update(self) -> None:
        max_norm = self.clip_norm if self.clip_norm is not None else math.inf
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

        # Bad step: the gradient blew up, or every table had a NaN/inf loss
        # (then there is no gradient at all, e.g. because the weights broke).
        all_losses_bad = self._nonfinite_in_step == self.accum_steps
        if torch.isfinite(grad_norm) and not all_losses_bad:
            self.optimizer.step()
            self._bad_in_a_row = 0
            self._record(grad_norm=grad_norm.item(), updates=1)
        else:
            self.skipped_steps += 1
            self._bad_in_a_row += 1
            if self._bad_in_a_row >= self.max_bad_steps:
                raise RuntimeError(
                    f"{self._bad_in_a_row} optimizer steps in a row had a "
                    "non-finite loss or gradient; stopping (lower the learning rate?)"
                )

        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1
        self._pending = 0
        self._nonfinite_in_step = 0

    def window_metrics(self) -> dict[str, float]:
        """Averages since the last call, then starts a new window."""
        w, self._window = self._window, defaultdict(float)
        elapsed = time.perf_counter() - self._window_start
        self._window_start = time.perf_counter()

        tables = max(w["tables"], 1)
        losses = max(w["losses"], 1)
        metrics = {
            "train/loss": w["loss"] / losses if w["losses"] else math.nan,
            "train/accuracy": w["accuracy"] / losses if w["losses"] else math.nan,
            "train/lr": self.optimizer.param_groups[0]["lr"],
            "train/grad_norm": w["grad_norm"] / max(w["updates"], 1),
            "train/skipped_steps": self.skipped_steps,
            "train/nonfinite_losses": w["nonfinite_losses"],
            "prior/n_features": w["n_features"] / tables,
            "prior/n_rows": w["n_rows"] / tables,
            "prior/n_classes": w["n_classes"] / tables,
            "time/data": w["data_time"],
            "time/compute": w["compute_time"],
            "time/train_total": self.train_time,
            "speed/tables_per_sec": w["tables"] / elapsed if elapsed else math.nan,
            "progress/tables_seen": self.tables_seen,
            "progress/epoch": self.epoch,
        }
        if self.device.type == "cuda":
            metrics["gpu/mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
        return metrics

    def train_epoch(
        self,
        tables: Iterator[TableSample],
        n_steps: int,
        logger: Logger | None = None,
        checkpointer: Any = None,
        log_every: int = 10,
    ) -> dict[str, float]:
        """Take n_steps optimizer steps; returns the mean loss and accuracy.

        The prior is an endless stream, so an epoch is a fixed number of
        steps. Stops early (after a whole step) if a stop was requested.
        """
        self.model.train()
        self._set_optimizer_mode(train=True)
        totals = {"loss": 0.0, "accuracy": 0.0, "tables": 0}

        progress = tqdm(
            total=n_steps, desc=f"epoch {self.epoch + 1}", leave=False, disable=None
        )
        target_step = self.step + n_steps
        while self.step < target_step and not self.stop_requested:
            start = time.perf_counter()
            batch = next(tables)
            self._record(data_time=time.perf_counter() - start)

            loss = self.train_step(batch)
            self.train_time += time.perf_counter() - start
            if loss is not None:
                totals["loss"] += loss
                totals["accuracy"] += self.last_accuracy
                totals["tables"] += 1

            if self._pending == 0:  # an update just happened
                progress.update()
                if logger is not None and self.step % log_every == 0:
                    logger.log(self.window_metrics(), step=self.step)
                if checkpointer is not None:
                    checkpointer.maybe_save(self)
        progress.close()

        n = max(totals["tables"], 1)
        return {"loss": totals["loss"] / n, "accuracy": totals["accuracy"] / n}

    def fit(
        self,
        tables: Iterable[TableSample],
        n_epochs: int,
        steps_per_epoch: int,
        evaluator: Any = None,
        logger: Logger | None = None,
        checkpointer: Any = None,
        log_every: int = 10,
    ) -> str:
        """Train until n_epochs are done; continues where a resumed run stopped.

        After every epoch: evaluate, log, save last.pt and possibly best.pt.
        Returns "finished", or "stopped" if a signal asked to stop early.
        """
        tables = iter(tables)
        while self.epoch < n_epochs:
            # A resumed run may stop in the middle of an epoch.
            remaining = (self.epoch + 1) * steps_per_epoch - self.step
            stats = self.train_epoch(tables, remaining, logger, checkpointer, log_every)
            if self.stop_requested:
                if checkpointer is not None:
                    checkpointer.save(self)
                print(f"Stopped at step {self.step}; resume from the checkpoint.")
                return "stopped"
            self.epoch += 1

            record = {
                "progress/epoch": self.epoch,
                "epoch/loss": stats["loss"],
                "epoch/accuracy": stats["accuracy"],
            }
            if evaluator is not None:
                with self.inference():
                    result = evaluator.evaluate()
                record.update({f"eval/{k}": v for k, v in result["summary"].items()})
                if logger is not None:
                    logger.log_datasets(result["datasets"], step=self.step)
            if logger is not None:
                logger.log(record, step=self.step)
            print(
                f"Epoch {self.epoch}/{n_epochs} | step {self.step} | "
                + " | ".join(f"{k} {v:.4g}" for k, v in record.items())
            )

            if checkpointer is not None:
                checkpointer.save(self)
                checkpointer.update_best(self, record)

        if checkpointer is not None:
            checkpointer.save_model(self)
        return "finished"

    # ----- checkpoints -----

    def state_dict(self) -> dict[str, Any]:
        np_state = np.random.get_state()
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None
            if self.scheduler is None
            else self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "tables_seen": self.tables_seen,
            "skipped_steps": self.skipped_steps,
            "train_time": self.train_time,
            # Restoring the generators makes a resumed run draw the same
            # random numbers in this process as an uninterrupted one would.
            "rng": {
                "python": random.getstate(),
                "numpy": (np_state[0], torch.from_numpy(np_state[1]), *np_state[2:]),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]
        self.epoch = state["epoch"]
        self.tables_seen = state["tables_seen"]
        self.skipped_steps = state["skipped_steps"]
        self.train_time = state["train_time"]

        rng = state["rng"]
        random.setstate(rng["python"])
        name, keys, *rest = rng["numpy"]
        np.random.set_state((name, keys.numpy(), *rest))
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
