"""Small model configuration, seeded tasks, and resumable trusted checkpoints."""

import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tabicl_reproduction.data import PriorDataset, TableSample
from tabicl_reproduction.model.model import NanoTabICLv2


@dataclass
class ModelConfig:
    max_classes: int = 10
    embed_dim: int = 32
    col_num_blocks: int = 1
    row_num_blocks: int = 1
    icl_num_blocks: int = 2
    col_nhead: int = 4
    row_nhead: int = 4
    icl_nhead: int = 4
    n_cls_cols: int = 2
    n_cls_rows: int = 16


class Model(NanoTabICLv2):
    """Assignment interface backed by the team's existing NanoTabICLv2."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        super().__init__(out_dim=self.config.max_classes, **asdict(self.config))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def isolated_seed(seed: int):
    state = rng_state()
    seed_all(seed)
    try:
        yield
    finally:
        restore_rng(state)


def valid_task(source: PriorDataset, max_attempts: int = 100) -> TableSample:
    for _ in range(max_attempts):
        sample = source.sample()
        if torch.unique(sample.y_train).numel() >= 2 and (sample.y_test >= 0).any():
            if not all(torch.isfinite(t).all() for t in sample):
                raise FloatingPointError("Prior generated non-finite values")
            return sample
    raise RuntimeError("Could not sample a nondegenerate task in 100 attempts")


def fixed_tasks(source: PriorDataset, count: int, seed: int) -> list[TableSample]:
    if count < 1:
        raise ValueError("count must be positive")
    with isolated_seed(seed):
        return [valid_task(source) for _ in range(count)]


def pack_tasks(tasks: list[TableSample]) -> list[dict]:
    return [
        dict(zip(("X_train", "y_train", "X_test", "y_test"), sample, strict=True))
        | {"dataset_id": sample.dataset_id}
        for sample in tasks
    ]


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Model, dict]:
    # RNG state includes NumPy objects. Load only checkpoints you trust.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = Model(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    return model.to(device), payload
