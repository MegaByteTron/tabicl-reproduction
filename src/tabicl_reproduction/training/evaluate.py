"""Evaluation entry point.

Locally:      python scripts/evaluate.py model=small checkpoint=<run dir>
On the GPU:   ... cfg/job=gpu_2h
The model config must be the one the run was trained with.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

from ml_project_template.config import run
from ml_project_template.utils import get_output_dir

from tabicl_reproduction.model import NanoTabICLv2
from tabicl_reproduction.training.checkpoint import load_model
from tabicl_reproduction.training.configs import Run
from tabicl_reproduction.training.evaluator import Evaluator
from tabicl_reproduction.training.metrics import MetricsLogger
from tabicl_reproduction.training.train import set_seed
from tabicl_reproduction.training.trainer import select_device


def main(
    cfg: Run,
    model: NanoTabICLv2,
    evaluator: Callable[..., Evaluator],
    checkpoint: str,
    weights: str = "best",
    device: str = "auto",
    out_dir: str | None = None,
) -> dict:
    """Score saved weights on the benchmark and write eval.json.

    checkpoint is a run directory (uses <weights>.pt: best or model) or a
    weights file.
    """
    out_dir_path = Path(out_dir) if out_dir is not None else get_output_dir()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    torch_device = select_device(device)
    step = load_model(model, checkpoint, weights)
    model.to(torch_device)

    result = evaluator(model, torch_device).evaluate()
    summary = result["summary"]
    (out_dir_path / "eval.json").write_text(
        json.dumps(
            {"checkpoint": checkpoint, "step": step, **result},
            indent=2,
        )
    )
    # Same metric names as during training, so W&B plots line up.
    MetricsLogger(out_dir_path).log(
        {f"eval/{k}": v for k, v in summary.items()}, step=step
    )
    print(f"Evaluated {checkpoint} (step {step}): {summary}")
    print(f"Writing results to {out_dir_path}")
    return result


def cli() -> None:
    import tabicl_reproduction.cluster.jobs  # noqa: F401  (cfg/job presets)
    from tabicl_reproduction.training import stores  # noqa: F401

    # run() always composes the "root" config; -cn picks the evaluation one.
    # Set before the job is submitted, so a cfg/job resubmission keeps it.
    sys.argv[1:1] = ["-cn", "eval_root"]
    run(main, seed_fn=set_seed)
