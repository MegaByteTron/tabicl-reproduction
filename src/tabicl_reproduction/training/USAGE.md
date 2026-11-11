# Training: all options

Every option is a Hydra override on the command line, picked as `group=name`
(choose a preset) or `key=value` (change one value). Run everything from the
repo root. The short guide is in [README.md](README.md); this page lists every
option.

```bash
uv run python scripts/train.py [overrides ...]
```

Add `--cfg job` to any command to print the fully resolved config instead of
training. Use it to check what an override does and to see every default.

```bash
uv run python scripts/train.py model=tiny optimizer=muon --cfg job
```

## Quick recipes

`...` stands for the command above (`uv run python scripts/train.py`).

```bash
# Smoke test (seconds): tiny model, no benchmark evaluation
... model=tiny epochs=2 steps_per_epoch=3

# Local run: small model, evaluated on tabpfn_val after every epoch
... model=small evaluator=tabpfn_val epochs=5

# TabICLv2 recipe: Muon + warmup/cosine schedule
... model=small optimizer=muon scheduler=warmup_cosine evaluator=tabpfn_val

# Plain AdamW on CPU in full precision, fixed output folder
... model=tiny optimizer=adamw device=cpu precision=fp32 out_dir=outputs/my_run

# Continue a run (see "Resume" below)
... model=tiny epochs=4 resume=outputs/2026-11-10/21-35-02
```

## Top-level options

| Option | Default | Meaning |
|---|---|---|
| `epochs` | `10` | Number of epochs. |
| `steps_per_epoch` | `100` | Optimizer steps per epoch. An evaluation and a checkpoint follow every epoch. |
| `accum_steps` | `8` | Tables per optimizer step. Each forward pass sees one table; the gradients of `accum_steps` tables are averaged into one update. |
| `clip_norm` | `1.0` | Gradient-norm clipping. Use `clip_norm=null` to turn it off. |
| `device` | `auto` | `auto` (CUDA, then MPS, then CPU), or `cpu`, `cuda`, `mps`. |
| `precision` | `auto` | `auto` (bf16 on GPUs that support it, else fp32), `bf16` or `fp32`. |
| `log_every` | `10` | Log training metrics to `metrics.jsonl` every this many steps. |
| `checkpoint_every_min` | `20.0` | Also save `last.pt` every this many minutes. `last.pt` is always saved after each epoch. |
| `monitor` | `eval/log_loss` | Metric that decides `best.pt` (lower is better). Needs an evaluator. |
| `resume` | `null` | Run folder (or `last.pt`) to continue from. |
| `out_dir` | `null` | Output folder. By default Hydra's `outputs/<date>/<time>/`. |
| `cfg.seed` | `0` | Seed for Python, numpy and torch. Written to `run.json`. |

The criterion is `CrossEntropyLoss(ignore_index=-1)`; `-1` marks a test row
whose class never appears in the training rows.

## Presets (`group=name`)

### `model=` (default `small`)

| Preset | `embed_dim` | `col/row/icl_num_blocks` | `col/row/icl_nhead` | Use |
|---|---|---|---|---|
| `paper` | 128 | 3 / 3 / 12 | 8 / 8 / 8 | Original TabICLv2 size |
| `small` | 96 | 2 / 2 / 4 | 4 / 4 / 4 | Experiments that fit in hours |
| `tiny` | 16 | 1 / 1 / 1 | 2 / 2 / 2 | Tests and smoke runs only |

Every preset has the same remaining defaults (`max_classes=10`,
`feature_group_size=3`, `n_cls_cols=4`, `n_cls_rows=128`). Override any value
with `model.<key>=...`, for example `model.embed_dim=32`. `max_classes` also
sets the prior's and the evaluator's class limit.

### `optimizer=` (default `adamw_sf`)

| Preset | What it is | Defaults |
|---|---|---|
| `adamw_sf` | AdamW schedule-free | `lr=4e-3`, `betas=[0.9, 0.999]`, `eps=1e-8`, `weight_decay=0.0`, `warmup_steps=0`, `r=0.0`, `weight_lr_power=2.0`, `foreach=true` |
| `adamw` | `torch.optim.AdamW` | `lr=1e-4`, `betas=[0.9, 0.999]`, `eps=1e-8`, `weight_decay=0.01`, `amsgrad=false` |
| `muon` | Muon on 2-D weight matrices, AdamW on everything else | `lr=8e-4`, `weight_decay=0.01`, `momentum=0.9`, `betas=[0.9, 0.999]` |

Override with `optimizer.<key>=...`, e.g. `optimizer.lr=1e-3`.

### `scheduler=` (default `constant`)

| Preset | What it is | Defaults |
|---|---|---|
| `constant` | Keeps the optimizer's learning rate (use with `adamw_sf`) | none |
| `warmup_cosine` | Linear warmup, then one cosine decay to `lr_end` | `warmup_proportion=0.01`, `lr_end=1e-7` |

The schedule covers `epochs * steps_per_epoch` steps.

### `data=` (default `prior`)

The training tables come from the synthetic prior, an endless stream.

| Option | Default | Meaning |
|---|---|---|
| `data.n_samples` | `1024` | Rows per table. |
| `data.n_train` | `0.8` | Training share of the rows (a float is a fraction, an int is a row count). |
| `data.min_features` / `data.max_features` | `2` / `100` | Range for the number of features per table. |
| `data.max_classes` | `${model.max_classes}` | Maximum number of classes per table. |
| `data.num_workers` | `$SLURM_CPUS_PER_TASK`, else `0` | Processes that generate tables in parallel. |
| `data.prefetch_factor` | `4` | Tables each worker prepares ahead. |

### `evaluator=` (default `none`)

| Preset | Meaning |
|---|---|
| `none` | No evaluation. No `eval/*` metrics, and `best.pt` is never written. |
| `tabpfn_val` | Scores the model on the real `tabpfn_val` OpenML tables after every epoch. Tables with more classes than `max_classes` are skipped. |

The first evaluation downloads the datasets, which takes a while. Each one is
stored by name, as `<folder>/<name>/features.parquet` and `target.parquet`. The
folder is `$TABICL_DATASETS_DIR` if set, else
`~/.cache/tabicl-reproduction/datasets/` inside a SLURM job, else
`.cache/datasets/`.

### `cfg/wandb=` (default off)

`cfg/wandb=base` logs the same metrics to Weights & Biases (the template reads
`WANDB_API_KEY`, `WANDB_ENTITY` and `WANDB_PROJECT` from the environment or a
`.env` file). Options: `cfg.wandb.mode=online|offline`, `cfg.wandb.group=...`.

### `cfg/job=` (default off)

`cfg/job=<preset>` submits the same command to the SLURM cluster instead of
running it locally. Presets: `test`, `gpu_2h`, `gpu_5h`, `sweep`. See
[cluster/README.md](../../../cluster/README.md) for what they mean and how to
run on the cluster.

## What a run writes

Everything goes to one folder per run, `outputs/<date>/<time>/` (or `out_dir`):

| File | When | Contents |
|---|---|---|
| `last.pt` | After every epoch, every `checkpoint_every_min` minutes, and on a stop | Full trainer state, used to resume. Overwritten each time. |
| `best.pt` | After an epoch whose `monitor` value is the best so far | Model weights only. Needs an evaluator. |
| `model.pt` | Once, when training finishes | Final model weights only. |
| `metrics.jsonl` | Every `log_every` steps, at each epoch end, and on a stop | One JSON row per log call. |
| `eval_datasets.jsonl` | After every evaluation | One row per dataset (accuracy, balanced accuracy, log loss, ROC AUC, ECE, time). |
| `run.json` | Written at the start, updated at the end | Commit, seed, device, hostname, `resumed_from`, and `end_time`, `status`, `step`, `epoch`, `signal`. |
| `.hydra/` | At the start | The resolved config and the overrides used. |

Read the logs with pandas:

```python
import pandas as pd

metrics = pd.read_json("outputs/<date>/<time>/metrics.jsonl", lines=True)
datasets = pd.read_json("outputs/<date>/<time>/eval_datasets.jsonl", lines=True)
```

`metrics.jsonl` mixes row types: training windows (`train/*`, `time/*`,
`speed/*`, `prior/*`), epoch summaries (`epoch/*` plus `eval/*` when an
evaluator is on), and a final `event: stopped` row after a stop. A column is
simply missing in rows of the other types.

## Stopping and resuming

- **Stop:** Ctrl-C, SIGTERM, and SLURM's SIGUSR2 (sent shortly before a time
  limit) all stop training after the current step. The run saves `last.pt`,
  writes a final `metrics.jsonl` row (`event: stopped`, `signal`,
  `progress/epoch`) and sets `status: stopped` in `run.json`.
- **Resume:** pass the old run folder. The model, optimizer, scheduler, step,
  epoch, random state and total training time are restored.

  ```bash
  ... model=small epochs=10 resume=outputs/2026-11-10/21-35-02
  ```

  The new run writes to a new timestamped folder, and its `run.json` records
  `resumed_from`. Pass the old folder as `out_dir` too to keep writing into it
  (this overwrites its `run.json`, `last.pt`, `best.pt` and `model.pt`). Use
  the same model, optimizer and scheduler options as the original run, and an
  `epochs` value larger than the epochs already done.
- **Status values in `run.json`:** `finished`, or `stopped`. A run without a
  `status` crashed or was killed without a signal.

## Metrics worth knowing

| Metric | Meaning |
|---|---|
| `train/loss`, `train/accuracy` | Training loss and accuracy on the prior's test rows, averaged over the window. |
| `epoch/loss`, `epoch/accuracy` | Same, averaged over the epoch. |
| `eval/accuracy`, `eval/balanced_accuracy`, `eval/log_loss`, `eval/roc_auc`, `eval/ece` | Mean over the `tabpfn_val` datasets (real data). ROC AUC is one-vs-rest. |
| `eval/n_datasets`, `eval/time_s` | Datasets evaluated and evaluation time. |
| `time/data`, `time/compute`, `time/train_total` | Seconds waiting for tables, seconds in forward/backward, and total training seconds (kept across resumes). |
| `speed/tables_per_sec` | Training throughput. |
| `train/grad_norm`, `train/lr`, `train/skipped_steps` | Gradient norm before clipping, current learning rate, and updates skipped because of a non-finite gradient. |
