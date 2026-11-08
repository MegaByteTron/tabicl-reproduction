"""Pretrain on the supplied prior; evaluate fixed tasks and save resumable state."""

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from tabicl_reproduction.data import PriorDataset, TableSample, to_model_inputs
from tabicl_reproduction.evaluation import (
    benchmark_tasks,
    evaluate_tasks,
    write_results,
)
from tabicl_reproduction.experiment import (
    Model,
    ModelConfig,
    atomic_save,
    fixed_tasks,
    load_checkpoint,
    pack_tasks,
    restore_rng,
    rng_state,
    seed_all,
    valid_task,
)


def task_loss(model: Model, sample: TableSample, device: str) -> torch.Tensor:
    X, labels = to_model_inputs(sample)
    n_classes = int(sample.y_train.max()) + 1
    if n_classes > model.config.max_classes:
        raise ValueError("Task exceeds model class capacity")
    target = sample.y_test.to(device)
    if not (target >= 0).any():
        raise ValueError("Task has no test labels observed in training")
    logits = model(X.to(device), labels.to(device))[0, :, :n_classes]
    return F.cross_entropy(logits, target, ignore_index=-1)


def training_step(
    model: Model,
    optimizer: torch.optim.Optimizer,
    tasks: list[TableSample],
    device: str,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    # Gradient accumulation permits different row/column counts per table.
    for sample in tasks:
        loss = task_loss(model, sample, device)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        (loss / len(tasks)).backward()
        total += loss.detach().item()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip, error_if_nonfinite=True
    )
    optimizer.step()
    return total / len(tasks)


def run(args: argparse.Namespace) -> None:
    for key in (
        "steps",
        "tables_per_step",
        "eval_every",
        "save_every",
        "val_tasks",
        "threads",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if (
        args.n_train < 2
        or args.n_test < 1
        or args.max_features < 2
        or args.max_classes < 2
    ):
        raise ValueError("Need >=2 training rows/features/classes and >=1 test row")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    history_path = args.output / "history.jsonl"
    if history_path.exists() and args.resume is None:
        raise ValueError(
            "Output already contains a run; choose a new directory or --resume"
        )
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    prior_config = {
        "n_train": args.n_train,
        "n_test": args.n_test,
        "max_features": args.max_features,
        "max_classes": args.max_classes,
    }
    start_step, best_loss, history = 0, float("inf"), []
    restored = None
    if args.resume:
        model, restored = load_checkpoint(args.resume, args.device)
        prior_config = restored["prior_config"]
        start_step = restored["step"]
        best_loss = restored["best_loss"]
        history = restored["history"]
        validation = [TableSample(**task) for task in restored["validation_tasks"]]
    else:
        model = Model(
            ModelConfig(max_classes=args.max_classes, embed_dim=args.embed_dim)
        )
        model.to(args.device)
        validation = fixed_tasks(
            PriorDataset(**prior_config), args.val_tasks, args.seed + 1
        )
    if start_step >= args.steps:
        raise ValueError("--steps is the total target and must exceed checkpoint step")
    source = PriorDataset(**prior_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if restored:
        optimizer.load_state_dict(restored["optimizer_state"])
        if args.tables_per_step != restored["training_config"]["tables_per_step"]:
            raise ValueError("Keep --tables-per-step unchanged when resuming")
        restore_rng(restored["rng"])
    real_tasks = []
    if args.real_data_dir:
        real_tasks, errors = benchmark_tasks(args.real_data_dir, args.real_ids)
        if errors or not real_tasks:
            (args.output / "real_load_errors.json").write_text(
                json.dumps(errors, indent=2)
            )
            raise RuntimeError(
                "Real validation data unavailable; run prepare-data first"
            )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "arguments": config,
                "model": asdict(model.config),
                "prior": prior_config,
                "git_revision": revision,
                "torch_version": str(torch.__version__),
                "real_validation_enabled": bool(real_tasks),
                "note": "Small classification prototype, AdamW; not full paper pretraining.",
            },
            indent=2,
        )
    )
    history_path.write_text("".join(json.dumps(row) + "\n" for row in history))

    def checkpoint(step: int, name: str):
        atomic_save(
            {
                "format_version": 1,
                "step": step,
                "model_config": asdict(model.config),
                "prior_config": prior_config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rng": rng_state(),
                "best_loss": best_loss,
                "history": history,
                "validation_tasks": pack_tasks(validation),
                "training_config": config,
            },
            args.output / name,
        )

    def evaluate(step: int) -> float:
        # sklearn/model evaluation must not change subsequent sampled training tasks.
        state = rng_state()
        try:
            rows = evaluate_tasks(model, validation, args.device, baselines=step == 0)
            result = write_results(rows, args.output / f"eval_prior_{step:06d}")
            metrics = result["methods"]["tabicl"]
            if metrics["n_failed"] or metrics["log_loss_known"] is None:
                raise RuntimeError(
                    "Prior validation failed; inspect evaluation outputs"
                )
            history.append({"step": step, "kind": "validation", **metrics})
            if real_tasks:
                real_rows = evaluate_tasks(
                    model, real_tasks, args.device, baselines=step == 0
                )
                write_results(real_rows, args.output / f"eval_real_{step:06d}")
                failures = [r for r in real_rows if r["status"] != "ok"]
                if failures:
                    raise RuntimeError(
                        "Real evaluation incomplete; inspect per_table.csv"
                    )
            print(
                f"step={step} validation_loss={metrics['log_loss_known']:.4f} "
                f"accuracy={metrics['accuracy']:.4f}",
                flush=True,
            )
            return metrics["log_loss_known"]
        finally:
            restore_rng(state)

    if not restored:
        best_loss = evaluate(0)
        checkpoint(0, "best.pt")
    if not real_tasks:
        print(
            "Real validation disabled: this run alone does NOT meet the full milestone.",
            flush=True,
        )
    start_time = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        loss = training_step(
            model,
            optimizer,
            [valid_task(source) for _ in range(args.tables_per_step)],
            args.device,
        )
        row = {
            "step": step,
            "kind": "train",
            "loss": loss,
            "seconds_since_start": time.perf_counter() - start_time,
        }
        history.append(row)
        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(step)
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint(step, "best.pt")
        if step % args.save_every == 0 or step == args.steps:
            checkpoint(step, f"step_{step:06d}.pt")
            checkpoint(step, "latest.pt")
        # Rewrite at evaluation/checkpoint boundaries, append otherwise.
        if step % args.eval_every == 0 or step == args.steps or step == start_step + 1:
            history_path.write_text("".join(json.dumps(r) + "\n" for r in history))
        else:
            with history_path.open("a") as file:
                file.write(json.dumps(row) + "\n")
        if step % 10 == 0:
            print(f"step={step} train_loss={loss:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--tables-per-step", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=10)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-tasks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--real-data-dir", type=Path)
    parser.add_argument("--real-ids", type=int, nargs="+")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
