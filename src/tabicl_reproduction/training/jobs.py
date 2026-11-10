"""SLURM job presets, selected on the command line with `cfg/job=<name>`.

Importing this module registers the presets in the `cfg/job` config group.
"""

from hydra_zen import builds, store
from ml_project_template.runs import Job, SlurmParams, SweepJob

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
_SHORT = {"test": ("gpu-test", 15), "gpu_9m": ("gpu-9m", 9)}
_LONG = {"gpu_2h": ("gpu-2h", 2), "gpu_5h": ("gpu-5h", 5)}

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
        builds(Job, image=IMAGE, slurm_params=_slurm_cfg, timeout_min=_minutes),
        name=_name,
    )

_sweep_slurm, _sweep_minutes = PRESETS["gpu_2h"]
job_store(
    builds(
        SweepJob,
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
