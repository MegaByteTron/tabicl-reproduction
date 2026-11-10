"""Check the SLURM presets without instantiating them.

Instantiating a Job submits it to SLURM and exits the process, so these tests
only look at the stored configs.
"""

import pytest
from hydra_zen import store
from omegaconf import OmegaConf

import tabicl_reproduction.training.jobs  # noqa: F401  (registers the presets)

# preset -> (partition, limit in minutes)
LIMITS = {
    "test": ("gpu-test", 15),
    "gpu_9m": ("gpu-9m", 9),
    "gpu_2h": ("gpu-2h", 120),
    "gpu_5h": ("gpu-5h", 300),
    "sweep": ("gpu-2h", 120),
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
