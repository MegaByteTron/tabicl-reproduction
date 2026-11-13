"""Check the SLURM presets without instantiating them.

Instantiating a Job submits it to SLURM and exits the process, so these tests
only look at the stored configs.
"""

import os
import pickle
import signal
import sys
import threading
import time

import pytest
from hydra_zen import instantiate, store
from omegaconf import OmegaConf
from submitit import AutoExecutor

from tabicl_reproduction.cluster.jobs import (
    DatasetSweepJob,
    StoppableCommand,
    StoppableJob,
)

# preset -> (partition, limit in minutes)
LIMITS = {
    "test": ("gpu-teaching-9m", 9),
    "gpu_2h": ("gpu-teaching-2h", 120),
    "gpu_5h": ("gpu-teaching-5h", 300),
    "sweep": ("gpu-teaching-2h", 120),
}


def load(name: str):
    return OmegaConf.structured(store["cfg/job", name])


@pytest.mark.parametrize("name", LIMITS)
def test_partition_and_time(name):
    cfg = load(name)
    partition, minutes = LIMITS[name]
    assert cfg.slurm_params.partition == partition
    assert cfg.slurm_params.gpus_per_node == 1
    assert cfg.timeout_min == minutes
    # time_hours becomes a "H:00:00" string that breaks submitit's time-limit
    # handler, so the limit is set through timeout_min only.
    assert cfg.slurm_params.get("time_hours") is None


@pytest.mark.parametrize("name", LIMITS)
def test_submitit_gets_the_time_limit_as_minutes(name, tmp_path, monkeypatch):
    # submitit computes with this value when the time-limit signal arrives
    # (job 5217341 crashed there with "2:00:00"). It needs srun to exist.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "srun").write_text("#!/bin/sh\n")
    (fake_bin / "srun").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    cfg = load(name)
    executor = AutoExecutor(folder=tmp_path / "jobs", cluster="slurm")
    executor.update_parameters(
        timeout_min=cfg.timeout_min,
        **instantiate(cfg.slurm_params).to_submitit_params(),
    )
    assert executor._executor.parameters["time"] == LIMITS[name][1]


@pytest.mark.parametrize("name", LIMITS)
def test_image_and_cpus(name):
    cfg = load(name)
    assert "megabytetron/tabicl-reproduction" in cfg.image
    assert cfg.slurm_params.cpus_per_task == (4 if LIMITS[name][1] < 60 else 8)


@pytest.mark.parametrize("name", ["gpu_2h", "gpu_5h", "sweep"])
def test_long_jobs_use_big_gpus(name):
    assert load(name).slurm_params.constraint == "80gb|40gb|h100|h200"


def test_sweep_settings():
    cfg = load("sweep")
    assert cfg.num_workers == 2
    assert cfg.metric_name == "eval/log_loss"
    assert cfg.metric_goal == "minimize"
    assert cfg.parameters == {"optimizer.lr": [1e-3, 4e-3]}


@pytest.mark.parametrize("name", ["test", "gpu_2h", "gpu_5h"])
def test_training_presets_use_the_stoppable_job(name):
    assert load(name)._target_.endswith(".StoppableJob")


def test_sweep_preset_mounts_the_datasets():
    assert load("sweep")._target_.endswith(".DatasetSweepJob")


@pytest.mark.parametrize("job_class", [StoppableJob, DatasetSweepJob])
def test_job_overrides_are_not_passed_to_the_job(job_class):
    # The job's own run has no cfg.job, so Hydra rejects any cfg.job override.
    job = object.__new__(job_class)
    args = [
        "scripts/train.py",
        "cfg/job=test",
        "cfg.job.slurm_params.cpus_per_task=8",
        "+cfg.job.timeout_min=5",
        "cfg/wandb=base",
        "cfg.wandb.mode=offline",
        "model=small",
    ]
    assert job.filter_args(args) == [
        "scripts/train.py",
        "cfg/wandb=base",
        "cfg.wandb.mode=offline",
        "model=small",
    ]


@pytest.mark.parametrize("job_class", [StoppableJob, DatasetSweepJob])
def test_job_container_mounts_the_shared_datasets(job_class, monkeypatch):
    monkeypatch.delenv("JOB_REPO", raising=False)
    # Creating a Job submits it, so build one without running __post_init__.
    job = object.__new__(job_class)
    job.image = "image.sif"
    command = job.python_command
    assert command.startswith("apptainer run -B /home/space/datasets ")
    assert command.endswith("--nv image.sif python")


# A stand-in for the training: on SIGUSR2 it finishes its step, then saves.
TRAINING = """
import pathlib, signal, sys, time
ready, saved = map(pathlib.Path, sys.argv[1:])
stop = False
def request_stop(*_):
    global stop
    stop = True
signal.signal(signal.SIGUSR2, request_stop)
ready.touch()
while not stop:
    time.sleep(0.05)
time.sleep(0.5)  # finish the current step
saved.touch()
"""


def test_time_limit_signal_waits_until_the_training_saved(tmp_path):
    ready, saved = tmp_path / "ready", tmp_path / "saved"
    command = StoppableCommand([sys.executable, "-c", TRAINING, str(ready), str(saved)])
    saved_when_checkpoint_returned = []

    # Like submitit: SIGUSR2 to this process calls checkpoint(), and the job
    # ends as soon as checkpoint() returns.
    def on_signal(*_):
        command.checkpoint()
        saved_when_checkpoint_returned.append(saved.exists())

    def send_time_limit_signal():
        while not ready.exists():
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGUSR2)

    previous = signal.signal(signal.SIGUSR2, on_signal)
    try:
        threading.Thread(target=send_time_limit_signal, daemon=True).start()
        assert command() == 0
    finally:
        signal.signal(signal.SIGUSR2, previous)
    assert saved_when_checkpoint_returned == [True]


def test_failing_training_fails_the_job():
    command = StoppableCommand([sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(RuntimeError, match="code 3"):
        command()


def test_command_can_be_pickled_for_submission():
    # submitit pickles the function before the job starts.
    command = StoppableCommand(["python", "scripts/train.py"])
    assert pickle.loads(pickle.dumps(command)).command == command.command
