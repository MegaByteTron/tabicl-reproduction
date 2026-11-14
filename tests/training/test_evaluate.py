import json
from functools import partial

import pytest
import torch

from tabicl_reproduction.training import Checkpointer, Evaluator, load_model
from tabicl_reproduction.training.configs import Run
from tabicl_reproduction.training.evaluate import main

from .conftest import tiny_model


@pytest.fixture
def run_dir(make_trainer, tables, tmp_path):
    trainer = make_trainer()
    trainer.fit(
        tables(), n_epochs=1, steps_per_epoch=2, checkpointer=Checkpointer(tmp_path)
    )
    return tmp_path


def test_load_model_restores_weights(run_dir):
    model = tiny_model()
    step = load_model(model, run_dir, weights="model")
    saved = torch.load(run_dir / "model.pt", weights_only=True)
    assert step == 2
    for name, value in saved["model"].items():
        assert torch.equal(model.state_dict()[name], value)


def test_main_writes_eval_json(run_dir, make_sample, tmp_path):
    out = tmp_path / "eval"
    evaluator = partial(Evaluator, tables=[make_sample(), make_sample()])
    result = main(
        Run(),
        tiny_model(),
        evaluator,
        checkpoint=str(run_dir),
        weights="model",
        device="cpu",
        out_dir=str(out),
    )

    saved = json.loads((out / "eval.json").read_text())
    assert saved["step"] == 2
    assert saved["summary"]["n_datasets"] == 2 == len(result["datasets"])
    assert (out / "metrics.jsonl").exists()
