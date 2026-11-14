"""SLURM job presets, selected on the command line with `cfg/job=<name>`.

Importing this module registers the presets in the `cfg/job` config group.
"""

import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any

from hydra_zen import builds, store
from ml_project_template.runs import Job, SlurmParams, SweepJob
from ml_project_template.utils import get_output_dir, logger
from submitit import AutoExecutor

from tabicl_reproduction.data.loader import CLUSTER_DATASETS_DIR


@dataclass
class ClusterJobMixin(Job):
    """Fixes for the template's jobs, shared by our training and sweep jobs.

    - Mounts the cluster's shared datasets folder in the job's container:
      apptainer only mounts the home directory by default, and
      /home/space/datasets is not inside it.
    - Drops `cfg.job.<key>=...` overrides (e.g. a different CPU count) from
      the command the job runs. The template drops only `cfg/job=...`, but
      the job's own run has no cfg.job, so Hydra would reject them.
    """

    def filter_args(self, args: list[str]) -> list[str]:
        args = super().filter_args(args)
        return [arg for arg in args if not arg.lstrip("+~").startswith("cfg.job.")]

    @property
    def python_command(self) -> str:
        command = super().python_command
        return command.replace(
            "apptainer run", f"apptainer run -B {CLUSTER_DATASETS_DIR}", 1
        )


class StoppableCommand:
    """Runs the training command; on the time limit or scancel, lets it save first.

    SLURM sends its signals to submitit's process, not to the training:
    SIGUSR2 before the time limit, SIGTERM on scancel (which submitit only
    logs). Both are forwarded, so the training saves last.pt and exits.
    """

    def __init__(self, command: list[str], stop_timeout_s: float = 60.0):
        self.command = command
        self.stop_timeout_s = stop_timeout_s
        self._process: subprocess.Popen | None = None
        self._waiter: threading.Thread | None = None

    def __call__(self) -> int:
        print(f"Running: {' '.join(self.command)}", flush=True)
        # Unbuffered, so the training's prints reach the job log even if it
        # is killed; its output goes straight to the log, not through a pipe.
        env = os.environ | {"PYTHONUNBUFFERED": "1"}
        self._process = subprocess.Popen(self.command, env=env)
        # Wait in a thread: the signal handler below runs in this thread and
        # must be able to wait for the process too.
        self._waiter = threading.Thread(target=self._process.wait, daemon=True)
        self._waiter.start()
        # Replaces submitit's handler, which only logs SIGTERM.
        signal.signal(signal.SIGTERM, self._on_cancel)
        while self._waiter.is_alive():
            self._waiter.join(timeout=1.0)
        if self._process.returncode != 0:
            raise RuntimeError(f"Training exited with code {self._process.returncode}")
        return self._process.returncode

    def checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Called by submitit on SIGUSR2. Returns None: no automatic requeue."""
        process, waiter = self._process, self._waiter
        if process is not None and waiter is not None and process.returncode is None:
            print("Time limit: asking the training to stop and save.", flush=True)
            process.send_signal(signal.SIGUSR2)
            # submitit ends the job when this returns, so wait for the save.
            waiter.join(timeout=self.stop_timeout_s)
        return None

    def _on_cancel(self, *args: Any) -> None:
        """scancel: __call__ keeps waiting while the training saves and exits."""
        if self._process is not None and self._process.returncode is None:
            print("Cancelled: asking the training to stop and save.", flush=True)
            self._process.send_signal(signal.SIGTERM)


@dataclass
class StoppableJob(ClusterJobMixin):
    """The template's Job, submitting a StoppableCommand.

    run() repeats ml_project_template's Job.run (pinned in pyproject.toml);
    only the submitted function differs. Check it when updating the template.
    """

    def run(self) -> None:
        hydra_run_dir = os.environ.get(
            "HYDRA_OUTPUT_DIR", "./outputs/runs/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}"
        )
        command = [
            "python",
            *self.filter_args(sys.argv),
            "cfg/wandb=base",
            f"hydra.run.dir={hydra_run_dir}",
        ]
        executor = AutoExecutor(
            folder=get_output_dir(),
            cluster=self.cluster,
            slurm_python=self.python_command,
        )
        executor.update_parameters(
            timeout_min=self.timeout_min,
            **self.slurm_params.to_submitit_params(),
        )
        job = executor.submit(StoppableCommand(command))
        logger.info(f"Submitted job {job.job_id}")
        if self.wait_for_job:
            logger.info(f"\n{job.result()}")


@dataclass
class DatasetSweepJob(ClusterJobMixin, SweepJob):
    """The template's SweepJob, with the shared datasets folder mounted."""


IMAGE = "${oc.env:JOB_IMAGE,oras://ghcr.io/megabytetron/tabicl-reproduction:latest-sif}"
# Any card with at least 40 GB, so jobs start on whichever node is free.
BIG_GPU = "80gb|40gb|h100|h200"


def _slurm(partition: str, cpus: int, **extra):
    """SLURM resources for one GPU on the given partition."""
    return builds(
        SlurmParams, partition=partition, gpus_per_node=1, cpus_per_task=cpus, **extra
    )


# Every preset sets the time limit only through `timeout_min` (the template's
# default is 5 minutes). Never set the template's `time_hours`: it becomes the
# string "2:00:00", which SLURM accepts but submitit then uses as its own time
# limit in minutes. Its handler for the time-limit signal multiplies that
# value, crashes, and the training is killed before it can save (job 5217341).
# Students may only use the GPU-TEACHING-* partitions (2D, 2H, 5H, 7D, 9M).
PRESETS = {
    "test": (_slurm("gpu-teaching-9m", 4), 9),
    "gpu_2h": (_slurm("gpu-teaching-2h", 8, constraint=BIG_GPU), 2 * 60),
    "gpu_5h": (_slurm("gpu-teaching-5h", 8, constraint=BIG_GPU), 5 * 60),
}

job_store = store(group="cfg/job")
for _name, (_slurm_cfg, _minutes) in PRESETS.items():
    job_store(
        builds(
            StoppableJob, image=IMAGE, slurm_params=_slurm_cfg, timeout_min=_minutes
        ),
        name=_name,
    )

_sweep_slurm, _sweep_minutes = PRESETS["gpu_2h"]
job_store(
    builds(
        DatasetSweepJob,
        image=IMAGE,
        slurm_params=_sweep_slurm,
        timeout_min=_sweep_minutes,
        num_workers=2,
        # Around 1e-3, the value that trained stably (4e-3 diverged).
        parameters={"optimizer.lr": [5e-4, 1e-3, 2e-3]},
        metric_name="eval/log_loss",
        metric_goal="minimize",
    ),
    name="sweep",
)
