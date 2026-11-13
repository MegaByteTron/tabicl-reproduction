import json
from pathlib import Path

import wandb

from tabicl_reproduction.training.metrics import MetricsLogger


class FakeRun:
    """Records what MetricsLogger sends to W&B, without a W&B account."""

    id = "abc123"

    def __init__(self):
        self.summary = {}
        self.saved = []
        self.artifacts = []

    def save(self, path, base_path, policy):
        self.saved.append(Path(path).relative_to(base_path).as_posix())

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


def fake_wandb(monkeypatch):
    run = FakeRun()
    finished = []
    monkeypatch.setattr(wandb, "run", run)
    monkeypatch.setattr(wandb, "finish", lambda: finished.append(True))
    return run, finished


def test_finish_writes_status_and_closes_the_wandb_run(tmp_path, monkeypatch):
    run, finished = fake_wandb(monkeypatch)
    logger = MetricsLogger(tmp_path)
    logger.write_run_info()

    logger.finish(status="stopped", signal="SIGUSR2")

    info = json.loads((tmp_path / "run.json").read_text())
    assert info["status"] == "stopped" and info["signal"] == "SIGUSR2"
    # The reason goes to the W&B summary, and we close the run ourselves
    # instead of leaving it to W&B's exit hook.
    assert run.summary == {"run/status": "stopped", "run/signal": "SIGUSR2"}
    assert finished == [True]


def test_finish_uploads_logs_and_models_but_not_last_pt(tmp_path, monkeypatch):
    run, _ = fake_wandb(monkeypatch)
    logger = MetricsLogger(tmp_path)
    logger.write_run_info()
    (tmp_path / "metrics.jsonl").write_text('{"step": 1}\n')
    for name in ("best.pt", "model.pt", "last.pt"):
        (tmp_path / name).write_bytes(b"weights")

    logger.finish(status="finished", step=1)

    # Only files that exist are uploaded; eval_datasets.jsonl was never written.
    assert run.saved == ["run.json", "metrics.jsonl"]
    (artifact,) = run.artifacts
    assert artifact.name == "model-abc123" and artifact.type == "model"
    assert artifact.metadata == {"status": "finished", "step": 1}
    assert sorted(artifact.manifest.entries) == ["best.pt", "model.pt"]


def test_a_failed_upload_still_finishes_the_run(tmp_path, monkeypatch):
    run, finished = fake_wandb(monkeypatch)

    def broken_save(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(run, "save", broken_save)
    logger = MetricsLogger(tmp_path)
    logger.write_run_info()
    logger.finish(status="finished")
    assert finished == [True]


def test_finish_without_wandb_only_writes_run_json(tmp_path, monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    logger = MetricsLogger(tmp_path)
    logger.write_run_info()
    logger.finish(status="finished", signal=None)
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "finished"
