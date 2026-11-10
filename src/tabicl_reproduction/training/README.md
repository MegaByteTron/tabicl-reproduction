# Training

Run from the repo root on your laptop. No SLURM, no W&B.

```bash
# Smoke test (seconds): tiny model, no benchmark evaluation (the default)
uv run python src/tabicl_reproduction/training/train.py \
  model=tiny epochs=2 steps_per_epoch=3

# A real local run: small model, evaluation on tabpfn_val after every epoch
uv run python src/tabicl_reproduction/training/train.py model=small evaluator=tabpfn_val epochs=5

# Other choices
uv run python src/tabicl_reproduction/training/train.py \
  model=small optimizer=muon scheduler=warmup_cosine

# Continue a run from its folder
uv run python src/tabicl_reproduction/training/train.py \
  model=tiny epochs=4 steps_per_epoch=3 resume=<run dir>
```

The run folder (printed as `Writing checkpoints and metrics to ...`, under
`outputs/`) contains `metrics.jsonl`, `eval_datasets.jsonl`, `run.json`,
`last.pt`, `best.pt` and `model.pt`. Read the metrics with
`pandas.read_json("<run dir>/metrics.jsonl", lines=True)`.
