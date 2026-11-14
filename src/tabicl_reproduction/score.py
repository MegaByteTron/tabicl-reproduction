"""Score a trained model on the same tables as the baselines.

From the repo root, with a run folder from training (it has .hydra/ and the
weights; see training/USAGE.md):

    uv run python -m tabicl_reproduction.score <run dir> --limit 5   # quick test
    uv run python -m tabicl_reproduction.score <run dir>             # best.pt
    uv run python -m tabicl_reproduction.score <run dir> --weights model

Writes results/tabicl_<weights>.csv next to the baselines, then prints the
summary of all models and saves results/summary.txt.
"""

import argparse
import time
from pathlib import Path

import torch
from hydra_zen import instantiate, load_from_yaml
from torch import nn

from tabicl_reproduction.baselines import result_tables, write_summary
from tabicl_reproduction.data import TableSample
from tabicl_reproduction.results import ResultsWriter, make_table_id
from tabicl_reproduction.training.checkpoint import load_model

N_ORDERS = 8  # the official TabICL code also averages 8 predictions


@torch.no_grad()
def predict_proba(
    model: nn.Module, table: TableSample, n_orders: int = N_ORDERS, seed: int = 0
) -> torch.Tensor:
    """Class probabilities for the test rows, averaged over column orders.

    The model depends on the order of the columns, so we predict with several
    random orders and average the probabilities. (The official TabICL code
    also shuffles columns over 8 predictions, plus other changes we leave out.)
    The first order is the original one, so n_orders=1 is a plain forward pass.
    """
    n_classes = int(table.y_train.max()) + 1
    X = torch.cat([table.X_train, table.X_test])
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    n_features = X.shape[1]
    orders = [torch.arange(n_features)]
    orders += [
        torch.randperm(n_features, generator=generator) for _ in range(n_orders - 1)
    ]

    proba = torch.zeros(len(table.X_test), n_classes)
    for order in orders:  # one at a time: a batch of 8 big tables needs a lot of memory
        x = X[:, order].unsqueeze(0).to(device)
        logits = model(x, table.y_train.unsqueeze(0).to(device))[0, :, :n_classes]
        proba += torch.softmax(logits.float(), dim=-1).cpu()
    return proba / len(orders)


def score(
    model: nn.Module,
    out_csv: Path,
    *,
    name: str,
    step: int,
    max_classes: int,
    limit: int | None = None,
    n_orders: int = N_ORDERS,
) -> None:
    """Score the model on every result table and write out_csv."""
    writer = ResultsWriter(out_csv, model=name, checkpoint_step=step)
    for info, tables in result_tables(limit):
        for table in tables:
            table_id = make_table_id(info["source"], table.dataset_id)
            n_classes = int(table.y_train.max()) + 1
            # One class gives a meaningless "perfect" score (the baselines skip
            # it too); more than max_classes can't be predicted.
            if not 2 <= n_classes <= max_classes:
                reason = f"{n_classes} classes, the model needs 2 to {max_classes}"
                writer.skip(table, reason, table_id=table_id, **info)
                continue
            start = time.perf_counter()
            proba = predict_proba(model, table, n_orders)
            time_s = time.perf_counter() - start
            writer.add(table, proba, table_id=table_id, time_s=time_s, **info)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a trained model.")
    parser.add_argument("run_dir", type=Path, help="training run folder")
    parser.add_argument("--weights", choices=["best", "model"], default="best")
    parser.add_argument("--limit", type=int, help="tables per source, quick test")
    parser.add_argument(
        "--out", type=Path, help="default: results/ (results_test/ with --limit)"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out = args.out or Path("results_test" if args.limit else "results")
    name = f"tabicl_{args.weights}"
    out_csv = out / f"{name}.csv"
    if out_csv.exists():
        # The writer appends, so a second run would add duplicate rows.
        raise SystemExit(f"{out_csv} already exists; move or delete it first")

    # The run saved its resolved config, so the model is built exactly as trained.
    config = load_from_yaml(args.run_dir / ".hydra" / "config.yaml")
    model = instantiate(config.model)
    step = load_model(model, args.run_dir, args.weights)
    model.to(args.device).eval()
    print(f"Scoring {args.run_dir / args.weights}.pt (step {step})")
    score(
        model,
        out_csv,
        name=name,
        step=step,
        max_classes=config.model.max_classes,
        limit=args.limit,
    )
    write_summary(out)


if __name__ == "__main__":
    main()
