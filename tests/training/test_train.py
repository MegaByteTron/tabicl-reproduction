import json
from functools import partial

import pandas as pd
import pytest
from hydra_zen import launch, store, zen
from hydra_zen.third_party.pydantic import pydantic_parser
from ml_project_template.config import pre_call

from tabicl_reproduction.training import stores  # noqa: F401  (registers configs)
from tabicl_reproduction.training.train import main, set_seed

TINY_RUN = [
    "model=tiny",
    "~evaluator",  # tabpfn_val would download from OpenML
    "epochs=2",
    "steps_per_epoch=2",
    "accum_steps=2",
    "data.n_samples=64",
    "data.max_features=4",
    "log_every=1",
]


@pytest.fixture(scope="module", autouse=True)
def hydra_store():
    store.add_to_hydra_store(overwrite_ok=True)


def run_cli(tmp_path, *overrides: str):
    # Same wrapping as the template's run(), without reading sys.argv.
    task = zen(
        main,
        pre_call=partial(pre_call, seed_fn=set_seed),
        resolve_pre_call=False,
        instantiation_wrapper=pydantic_parser,
    )
    return launch(
        store[None, "root"],
        task,
        overrides=[*TINY_RUN, f"hydra.run.dir={tmp_path}", *overrides],
        version_base=None,
    )


def test_cli_run_writes_metrics_checkpoints_and_config(tmp_path):
    run_cli(tmp_path)

    metrics = pd.read_json(tmp_path / "metrics.jsonl", lines=True)
    assert metrics["step"].max() == 4
    assert {"train/loss", "time/data", "epoch/loss"} <= set(metrics.columns)
    assert (tmp_path / "last.pt").exists() and (tmp_path / "model.pt").exists()
    assert (tmp_path / ".hydra" / "config.yaml").exists()
    assert json.loads((tmp_path / "run.json").read_text())["seed"] == 0


def test_cli_resume(tmp_path):
    run_cli(tmp_path / "first", "epochs=1")
    run_cli(tmp_path / "second", f"resume={tmp_path / 'first'}")

    metrics = pd.read_json(tmp_path / "second" / "metrics.jsonl", lines=True)
    assert metrics["step"].min() > 2  # continued after the first run's 2 steps


@pytest.mark.parametrize(
    "group", ["optimizer=adamw", "optimizer=muon", "scheduler=warmup_cosine"]
)
def test_every_option_runs(tmp_path, group):
    run_cli(tmp_path, group, "epochs=1")
    assert (tmp_path / "model.pt").exists()
