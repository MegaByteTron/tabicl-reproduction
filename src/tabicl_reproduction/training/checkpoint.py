import math
import signal
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .trainer import Trainer

# SLURM sends SIGUSR2 shortly before the time limit (submitit asks for it
# 90 s early) and SIGTERM when it kills the job; Ctrl-C sends SIGINT.
STOP_SIGNALS = (signal.SIGUSR2, signal.SIGTERM, signal.SIGINT)


def load_model(model: nn.Module, path: str | Path, weights: str = "best") -> int:
    """Load trained weights into model and return the step they were saved at.

    path is a weights file (best.pt, model.pt) or a run directory, in which
    case <weights>.pt is used. Needs no Trainer, so it also works for inference.
    """
    path = Path(path)
    if path.is_dir():
        path = path / f"{weights}.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return state["step"]


def _atomic_save(obj: Any, path: Path) -> None:
    # Write to a temporary file and rename, so a job that gets killed
    # mid-write never leaves a broken checkpoint.
    tmp_path = path.with_suffix(".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


class Checkpointer:
    """Saves everything needed to resume a run, plus the best and final weights.

    Files in directory:
    - last.pt: full trainer state (resume from it with resume=<run dir>)
    - best.pt: model weights with the best value of the monitored metric
    - model.pt: model weights at the end of training
    """

    def __init__(
        self,
        directory: str | Path,
        every_min: float = 20.0,
        monitor: str = "eval/log_loss",
        mode: str = "min",
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.every_s = every_min * 60
        self.monitor = monitor
        self.mode = mode
        self.best: float | None = None
        self._last_save = time.monotonic()

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    def save(self, trainer: Trainer) -> None:
        _atomic_save(
            {"trainer": trainer.state_dict(), "best": self.best}, self.last_path
        )
        self._last_save = time.monotonic()

    def maybe_save(self, trainer: Trainer) -> bool:
        """Save last.pt if every_min minutes passed since the last save."""
        if time.monotonic() - self._last_save < self.every_s:
            return False
        self.save(trainer)
        return True

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best

    def _save_weights(self, trainer: Trainer, name: str, **extra: Any) -> None:
        # Schedule-free optimizers keep the weights to evaluate apart from the
        # training weights; inference() swaps them in.
        with trainer.inference():
            state = {"model": trainer.model.state_dict(), "step": trainer.step}
            _atomic_save(state | extra, self.directory / name)

    def update_best(self, trainer: Trainer, metrics: dict[str, float]) -> bool:
        """Save best.pt if metrics[monitor] is the best value so far."""
        value = metrics.get(self.monitor)
        if value is None or math.isnan(value) or not self._improved(value):
            return False
        self.best = value
        self._save_weights(trainer, "best.pt", **{self.monitor: value})
        return True

    def save_model(self, trainer: Trainer) -> None:
        self._save_weights(trainer, "model.pt")

    def load(self, trainer: Trainer, path: str | Path) -> None:
        """Restore a run from last.pt, or from a run directory containing it."""
        path = Path(path)
        if path.is_dir():
            path = path / "last.pt"
        # Load to CPU: the RNG states must be CPU tensors; load_state_dict
        # copies the weights onto the trainer's device.
        state = torch.load(path, map_location="cpu", weights_only=True)
        trainer.load_state_dict(state["trainer"])
        self.best = state["best"]

    def install_signal_handlers(self, trainer: Trainer) -> None:
        """Ask the trainer to stop (and save) when SLURM or Ctrl-C ends the run."""

        def request_stop(signum: int, frame: Any) -> None:
            print(f"Received {signal.Signals(signum).name}, stopping after this step.")
            trainer.stop_requested = True
            trainer.stop_reason = signal.Signals(signum).name

        for sig in STOP_SIGNALS:
            signal.signal(sig, request_stop)
