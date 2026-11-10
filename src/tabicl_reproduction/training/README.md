# Training

Run from the repo root on your laptop. No SLURM, no W&B.

```bash
# Smoke test (seconds): tiny model, no benchmark evaluation
uv run python scripts/train.py \
  model=tiny '~evaluator' epochs=2 steps_per_epoch=3

# A real local run: small model, evaluation on tabpfn_val after every epoch
uv run python scripts/train.py model=small epochs=5

# Other choices
uv run python scripts/train.py \
  model=small optimizer=muon scheduler=warmup_cosine

# Continue a run from its folder
uv run python scripts/train.py \
  model=tiny '~evaluator' epochs=4 steps_per_epoch=3 resume=<run dir>
```

The run folder (printed as `Writing checkpoints and metrics to ...`, under
`outputs/`) contains `metrics.jsonl`, `eval_datasets.jsonl`, `run.json`,
`last.pt`, `best.pt` and `model.pt`. Read the metrics with
`pandas.read_json("<run dir>/metrics.jsonl", lines=True)`.

## Evaluation

Score saved weights on the benchmark, without training. Use the same model
config as the run (`model=small`); `weights=best` (default) or `weights=model`:

```bash
uv run python scripts/evaluate.py model=small checkpoint=<run dir>
```

It writes `eval.json` (summary and one row per dataset) to its own run folder.
`cfg/job=<preset>` submits it to the cluster like training.
