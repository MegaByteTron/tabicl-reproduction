import json
from types import SimpleNamespace

import wandb

from tabicl_reproduction.training.metrics import MetricsLogger


def test_finish_writes_status_and_closes_the_wandb_run(tmp_path, monkeypatch):
    run = SimpleNamespace(summary={})
    finished = []
    monkeypatch.setattr(wandb, "run", run)
    monkeypatch.setattr(wandb, "finish", lambda: finished.append(True))
    logger = MetricsLogger(tmp_path)
    logger.write_run_info()

    logger.finish(status="stopped", signal="SIGUSR2")

    info = json.loads((tmp_path / "run.json").read_text())
    assert info["status"] == "stopped" and info["signal"] == "SIGUSR2"
    # The reason goes to the W&B summary, and we close the run ourselves
    # instead of leaving it to W&B's exit hook.
    assert run.summary == {"run/status": "stopped", "run/signal": "SIGUSR2"}
    assert finished == [True]


def test_finish_without_wandb_only_writes_run_json(tmp_path, monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    logger = MetricsLogger(tmp_path)
    logger.write_run_info()
    logger.finish(status="finished", signal=None)
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "finished"
