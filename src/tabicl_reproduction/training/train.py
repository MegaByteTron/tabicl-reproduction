"""Training entry point.

Locally:      python scripts/train.py model=tiny
On the GPU:   ... cfg/job=gpu_2h          (submits a SLURM job, see cluster/)
Sweep:        ... cfg/job=sweep
Resume:       ... resume=<old run dir>
"""

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ml_project_template.config import run
from ml_project_template.utils import get_output_dir
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tabicl_reproduction.model import NanoTabICLv2
from tabicl_reproduction.training.checkpoint import Checkpointer
from tabicl_reproduction.training.configs import Run
from tabicl_reproduction.training.evaluator import Evaluator
from tabicl_reproduction.training.metrics import MetricsLogger
from tabicl_reproduction.training.trainer import Trainer


def set_seed(seed: int) -> None:
    # The prior draws from the global numpy and torch generators.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(
    cfg: Run,
    model: NanoTabICLv2,
    criterion: nn.Module,
    optimizer: Callable[..., Optimizer],
    scheduler: Callable[..., LRScheduler],
    # Any, not Iterable: the CLI's pydantic parser would call iter(data) before
    # this function runs, so DataLoader workers would be seeded before resume
    # restores the random state, and a resumed run would repeat its tables.
    data: Any,
    evaluator: Callable[..., Evaluator] | None = None,
    device: str = "auto",
    epochs: int = 10,
    steps_per_epoch: int = 100,
    accum_steps: int = 8,
    clip_norm: float | None = 1.0,
    precision: str = "auto",
    log_every: int = 10,
    checkpoint_every_min: float = 20.0,
    monitor: str = "eval/log_loss",
    resume: str | None = None,
    out_dir: str | None = None,
) -> Trainer:
    """Train a model on the prior; every argument is a config field.

    One epoch = steps_per_epoch optimizer steps of accum_steps tables each,
    followed by evaluation and a checkpoint.
    """
    out_dir_path = Path(out_dir) if out_dir is not None else get_output_dir()
    # Faster float32 matmuls on Ampere+ GPUs with negligible precision loss.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    built_optimizer = optimizer(model.parameters())
    trainer = Trainer(
        model,
        built_optimizer,
        criterion,
        device=None if device == "auto" else device,
        scheduler=scheduler(built_optimizer, total_steps=epochs * steps_per_epoch),
        accum_steps=accum_steps,
        clip_norm=clip_norm,
        precision=precision,
    )
    checkpointer = Checkpointer(out_dir_path, checkpoint_every_min, monitor=monitor)
    if resume is not None:
        checkpointer.load(trainer, resume)
        print(f"Resumed from {resume} at step {trainer.step}, epoch {trainer.epoch}")
    checkpointer.install_signal_handlers(trainer)

    logger = MetricsLogger(out_dir_path)
    logger.write_run_info(
        commit=cfg.commit,
        seed=cfg.seed,
        resumed_from=resume,
        device=str(trainer.device),
    )
    print(f"Writing checkpoints and metrics to {out_dir_path}")

    status = trainer.fit(
        data,
        n_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        evaluator=None if evaluator is None else evaluator(model, trainer.device),
        logger=logger,
        checkpointer=checkpointer,
        log_every=log_every,
    )
    logger.finish(
        end_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        status=status,
        step=trainer.step,
        epoch=trainer.epoch,
        signal=trainer.stop_reason,
    )
    return trainer


def cli() -> None:
    # Registering the configs is kept out of import time, as in the template.
    import tabicl_reproduction.cluster.jobs  # noqa: F401  (cfg/job presets)
    from tabicl_reproduction.training import stores  # noqa: F401

    run(main, seed_fn=set_seed)
