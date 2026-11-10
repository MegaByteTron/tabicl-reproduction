"""Registers all configs with Hydra; importing this module is enough.

Pick options with `group=name` (model=small, optimizer=muon) and change
single values with `key=value` (optimizer.lr=1e-3, epochs=20).
"""

from hydra_zen import MISSING, builds, store

from tabicl_reproduction.training import configs
from tabicl_reproduction.training.evaluate import main as evaluate_main
from tabicl_reproduction.training.train import main

MainConfig = builds(
    main,
    cfg=MISSING,
    criterion=configs.Criterion,
    populate_full_signature=True,
)
store(
    MainConfig,
    name="root",
    hydra_defaults=[
        "_self_",
        {"cfg": "base"},
        {"cfg/wandb": None},
        {"cfg/job": None},
        {"model": "small"},
        {"optimizer": "adamw_sf"},
        {"scheduler": "constant"},
        {"data": "prior"},
        {"evaluator": "tabpfn_val"},
    ],
)

# scripts/evaluate.py selects this root with `-cn eval_root`; it reuses the
# model and evaluator groups below, so `model=small` means the same in both.
EvalConfig = builds(
    evaluate_main,
    cfg=MISSING,
    populate_full_signature=True,
)
store(
    EvalConfig,
    name="eval_root",
    hydra_defaults=[
        "_self_",
        {"cfg": "base"},
        {"cfg/wandb": None},
        {"cfg/job": None},
        {"model": "small"},
        {"evaluator": "tabpfn_val"},
    ],
)

store(configs.RunConfig, group="cfg", name="base")
store(configs.WandBConfig, group="cfg/wandb", name="base")

model_store = store(group="model")
model_store(configs.PaperModel, name="paper")
model_store(configs.SmallModel, name="small")
model_store(configs.TinyModel, name="tiny")

optimizer_store = store(group="optimizer")
optimizer_store(configs.AdamWScheduleFree, name="adamw_sf")
optimizer_store(configs.AdamW, name="adamw")
optimizer_store(configs.Muon, name="muon")

scheduler_store = store(group="scheduler")
scheduler_store(configs.ConstantSchedule, name="constant")
scheduler_store(configs.WarmupCosine, name="warmup_cosine")

store(configs.PriorData, group="data", name="prior")
store(configs.TabPFNValEvaluator, group="evaluator", name="tabpfn_val")
