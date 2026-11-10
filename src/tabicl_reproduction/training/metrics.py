import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import torch
import wandb


class MetricsLogger:
    """Appends metrics to JSON-lines files in the run directory.

    - metrics.jsonl: one row per log call (train windows and epoch summaries)
    - eval_datasets.jsonl: one row per benchmark dataset per evaluation

    Read them with pd.read_json(path, lines=True). If a W&B run is active
    (cfg/wandb=base), the same metrics also go to W&B.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, rows: list[dict[str, Any]]) -> None:
        with (self.directory / name).open("a") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self._append("metrics.jsonl", [{"step": step, "time": time.time()} | metrics])
        if wandb.run is not None:
            wandb.log(metrics, step=step)

    def log_datasets(self, rows: list[dict[str, Any]], step: int) -> None:
        self._append("eval_datasets.jsonl", [{"step": step} | row for row in rows])

    def update_run_info(self, **extra: Any) -> None:
        """Add fields (end time, status) to run.json after training."""
        path = self.directory / "run.json"
        info = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps(info | extra, indent=2))

    def write_run_info(self, **extra: Any) -> dict[str, Any]:
        """Write run.json: where and with what this run was trained."""
        cuda = torch.cuda.is_available()
        info = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if cuda else None,
        } | extra
        (self.directory / "run.json").write_text(json.dumps(info, indent=2))
        return info
