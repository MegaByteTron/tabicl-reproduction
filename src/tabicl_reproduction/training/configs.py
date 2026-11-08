"""hydra-zen configs: every option you can pick on the command line.

Configs are generated from the real signatures with builds(), so a new
argument of NanoTabICLv2 or an optimizer is configurable automatically.
"""

from typing import NamedTuple

import schedulefree
import torch
from hydra_zen import builds
from ml_project_template.runs import Job
from ml_project_template.wandb import WandBRun
from torch import nn

from tabicl_reproduction.data import load_data
from tabicl_reproduction.model import NanoTabICLv2

from .evaluator import Evaluator
from .optim import constant_schedule, muon_adamw, warmup_cosine


class Run(NamedTuple):
    """Settings the cluster template reads before the run starts (cfg.*)."""

    seed: int | None = 0
    wandb: WandBRun | None = None
    job: Job | None = None
    commit: str | None = None  # filled in with the git commit by the template


# populate_full_signature: without it, seed would be missing and never set.
RunConfig = builds(Run, populate_full_signature=True, commit=None)
WandBConfig = builds(WandBRun, group=None, mode="online")

# ----- model (model=paper|small|tiny) -----

PaperModel = builds(
    NanoTabICLv2,
    populate_full_signature=True,
    max_classes=10,
    out_dim="${.max_classes}",  # classification: one logit per class
)
# Small size from the nanotabicl README, for experiments that fit in hours.
SmallModel = builds(
    NanoTabICLv2,
    embed_dim=96,
    col_num_blocks=2,
    row_num_blocks=2,
    icl_num_blocks=4,
    col_nhead=4,
    row_nhead=4,
    icl_nhead=4,
    builds_bases=(PaperModel,),
)
# Only for tests and smoke runs.
TinyModel = builds(
    NanoTabICLv2,
    embed_dim=16,
    col_num_blocks=1,
    row_num_blocks=1,
    icl_num_blocks=1,
    col_nhead=2,
    row_nhead=2,
    icl_nhead=2,
    builds_bases=(PaperModel,),
)

# ----- optimizer (optimizer=adamw_sf|adamw|muon) -----
# Partial configs: the training code calls them with the model's parameters.
# hydra_convert="all" passes plain lists/tuples (not OmegaConf objects), which
# end up in the optimizer state and must load from a checkpoint safely.

AdamWScheduleFree = builds(
    schedulefree.AdamWScheduleFree,
    zen_partial=True,
    hydra_convert="all",
    populate_full_signature=True,
    lr=4e-3,
    weight_decay=0.0,
)
AdamW = builds(
    torch.optim.AdamW,
    zen_partial=True,
    hydra_convert="all",
    populate_full_signature=True,
    lr=1e-4,  # TabICL's AdamW learning rate
    weight_decay=0.01,
)
Muon = builds(
    muon_adamw, zen_partial=True, hydra_convert="all", populate_full_signature=True
)

# ----- scheduler (scheduler=constant|warmup_cosine) -----

ConstantSchedule = builds(constant_schedule, zen_partial=True)
WarmupCosine = builds(warmup_cosine, zen_partial=True, populate_full_signature=True)

# ----- data and evaluation -----

Criterion = builds(nn.CrossEntropyLoss, ignore_index=-1)  # -1: class unseen in train

PriorData = builds(
    load_data,
    dataset="prior",
    n_train=0.8,
    n_samples=1024,
    min_features=2,
    max_features=100,
    max_classes="${model.max_classes}",
    # One prior worker per CPU that SLURM gives the job; 0 = no workers.
    num_workers="${oc.decode:${oc.env:SLURM_CPUS_PER_TASK,0}}",
    prefetch_factor=4,
)

TabPFNValEvaluator = builds(
    Evaluator,
    zen_partial=True,
    tables=builds(load_data, dataset="tabpfn_val"),
    max_classes="${model.max_classes}",
)
