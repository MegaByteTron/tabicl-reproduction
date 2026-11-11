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


class MountsDatasets:
    """Makes the cluster's shared datasets folder visible in the job's container.

    Apptainer only mounts the home directory by default, and
    /home/space/datasets is not inside it.
    """

    @property
    def python_command(self) -> str:
        command = super().python_command  # type: ignore[misc]
        return command.replace(
            "apptainer run", f"apptainer run -B {CLUSTER_DATASETS_DIR}", 1
        )


class StoppableCommand:
    """Runs the training command; on the time-limit signal, lets it save first.

    SLURM sends SIGUSR2 to submitit's process, not to the training. For a
    function without a checkpoint method, submitit then ends the job at once
    and the training is killed mid-step. This checkpoint method forwards the
    signal and waits until the training has saved last.pt and exited.
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
        while self._waiter.is_alive():
            self._waiter.join(timeout=1.0)
        if self._process.returncode != 0:
            raise RuntimeError(f"Training exited with code {self._process.returncode}")
        return self._process.returncode

    def checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Called by submitit on SIGUSR2. Returns None: no automatic requeue."""
        if self._process is not None and self._process.returncode is None:
            print("Time limit: asking the training to stop and save.", flush=True)
            self._process.send_signal(signal.SIGUSR2)
            self._waiter.join(timeout=self.stop_timeout_s)
        return None


@dataclass
class StoppableJob(MountsDatasets, Job):
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
class DatasetSweepJob(MountsDatasets, SweepJob):
    """The template's SweepJob, with the shared datasets folder mounted."""


IMAGE = "${oc.env:JOB_IMAGE,oras://ghcr.io/megabytetron/tabicl-reproduction:latest-sif}"
# Any card with at least 40 GB, so jobs start on whichever node is free.
BIG_GPU = "80gb|40gb|h100|h200"


def _slurm(partition: str, cpus: int, hours: int | None = None, **extra):
    """SLURM resources for one GPU on the given partition."""
    return builds(
        SlurmParams,
        partition=partition,
        gpus_per_node=1,
        cpus_per_task=cpus,
        time_hours=hours,
        **extra,
    )


# The template's default `timeout_min` is 5 minutes, so every preset sets its
# own. submitit lets `time_hours` (an int) override `timeout_min`, which is why
# sub-hour partitions set only `timeout_min`. Long presets set both to the same
# limit so the two never disagree.
# Students may only use the GPU-TEACHING-* partitions (2D, 2H, 5H, 7D, 9M).
_SHORT = {"test": ("gpu-teaching-9m", 9)}
_LONG = {"gpu_2h": ("gpu-teaching-2h", 2), "gpu_5h": ("gpu-teaching-5h", 5)}

PRESETS = {
    **{
        name: (_slurm(partition, 4), minutes)
        for name, (partition, minutes) in _SHORT.items()
    },
    **{
        name: (_slurm(partition, 8, hours, constraint=BIG_GPU), hours * 60)
        for name, (partition, hours) in _LONG.items()
    },
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
        parameters={"optimizer.lr": [1e-3, 4e-3]},
        metric_name="eval/log_loss",
        metric_goal="minimize",
    ),
    name="sweep",
)
