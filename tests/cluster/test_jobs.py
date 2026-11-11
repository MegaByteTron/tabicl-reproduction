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
from hydra_zen import store
from omegaconf import OmegaConf

from tabicl_reproduction.cluster.jobs import StoppableCommand

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
    hours = cfg.slurm_params.time_hours
    # time_hours overrides timeout_min in submitit, so it must agree if set.
    assert hours is None or hours * 60 == minutes
    # Sub-hour partitions cannot be expressed in whole hours.
    assert (hours is None) == (minutes < 60)


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
