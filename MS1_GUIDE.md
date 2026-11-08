# Milestone 1: training and evaluation

This branch starts from `origin/feat/visualization` at
`bb2f8e43572c1ff7953f127b70d42e015674691a`, which already includes
the data-loading work. It adds training, checkpoint/resume, a classifier wrapper,
same-split baselines, evaluation exports, and learning curves. No upstream PR has
been merged and nothing has been pushed to GitHub by this implementation task.

## What is implemented

| Assignment item | Location |
| --- | --- |
| Existing lazy prior / OpenML loading | `src/tabicl_reproduction/data/` |
| Existing visualization `show` | `src/tabicl_reproduction/viz/` |
| Small differentiable `Model` wrapper | `src/tabicl_reproduction/experiment.py` |
| Pretraining / validation / resume | `src/tabicl_reproduction/training.py` |
| Inference-only `Classifier` | `src/tabicl_reproduction/classifier.py` |
| Baselines and per-table metrics | `src/tabicl_reproduction/evaluation.py` |
| Benchmark caching | `src/tabicl_reproduction/prepare_data.py` |
| Loss and accuracy figure | `src/tabicl_reproduction/plot_training.py` |

The original architecture is retained. The small default has 32-dimensional
embeddings, 1 column block, 1 row block, 2 ICL blocks, and 16 inducing points.
AdamW is deliberately used for a simple prototype; this is not the paper's full
Muon pretraining recipe. Classification only; no many-class encoding wrapper.
Tables with more classes than the checkpoint supports produce explicit errors.

## Install

Run commands from the repository root. On the GPU cluster, follow the existing
project's locked environment:

```bash
uv sync --locked --group dev
```

For a CPU-only machine, this alternative avoids downloading CUDA libraries.
`--no-sync` below is important: a subsequent ordinary `uv run` may reinstall the
CUDA-enabled dependency from the lockfile.

```bash
uv venv --python 3.13
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy pandas matplotlib scipy scikit-learn openml pyarrow pytest ruff ty
uv pip install --no-deps -e .
```

## 1. Verify a short run

```bash
uv run --no-sync python -m tabicl_reproduction.training \
  --output outputs/first_run \
  --steps 100 --tables-per-step 2 \
  --n-train 64 --n-test 32 --max-features 4 --max-classes 2 \
  --val-tasks 8 --eval-every 25 --save-every 25 --device cpu
```

This binary, small-table run is an integration check, **not evidence of full
milestone completion**. The full requirement needs converged pretraining,
held-out-prior baseline comparisons, and periodic real-data evaluation.

Each optimizer step accumulates gradients over several independently generated
tables. Shapes may vary between tables. Training only passes `X` and `y_train`
to the model; `y_test` is used exclusively in the loss. Single-class training
tasks and tasks with no evaluable test labels are retried. Training data is
sampled continuously from the supplied actual prior, not a substitute generator.

## 2. Cache real development datasets

Use the cluster's assigned shared location (make sure you have write access):

```bash
uv run --no-sync python -m tabicl_reproduction.prepare_data \
  --data-dir /home/space/datasets/tabpfn_val
```

For a first connectivity check, add `--ids 59` (ionosphere). Omitting `--ids`
uses the existing team's registry in `data/benchmarks.py`. That registry currently
contains 57 IDs; the paper describes 60 development datasets. Verify the team's
list against Supplementary Table 5 and document any missing/unsupported datasets
before claiming full benchmark coverage. This implementation does not silently
invent the three additional IDs.

The existing downloader sets shared-read and directory traversal permissions.
It writes `download_manifest.json`, reports failures, and exits nonzero on
incomplete downloads. Training never silently downloads on a GPU allocation.

## 3. Run a larger experiment with real validation

```bash
uv run --no-sync python -m tabicl_reproduction.training \
  --output outputs/ms1_run \
  --steps 5000 --tables-per-step 8 \
  --n-train 128 --n-test 32 --max-features 10 --max-classes 10 \
  --val-tasks 32 --eval-every 250 --save-every 250 \
  --device cuda --threads 4 \
  --real-data-dir /home/space/datasets/tabpfn_val
```

This is an initial budget, not a promise of convergence. Examine curves and
independent evaluation before deciding whether more training is needed. Use
`--real-ids 59` only while debugging; report that subset explicitly.
Validation uses 80/20 real-data splits, caps training rows at 2048, and uses a
fixed seed. Small real datasets retain a useful train/test ratio. Test predictions
are chunked to limit memory, while retaining the entire training context.

If a real dataset is missing, training fails with an error report. If a model or
baseline cannot evaluate a table, its error is recorded; it is not silently
reported as a success. If real validation is omitted entirely, the console and
configuration clearly flag it.

## 4. Resume a checkpoint

```bash
uv run --no-sync python -m tabicl_reproduction.training \
  --output outputs/ms1_run \
  --resume outputs/ms1_run/latest.pt \
  --steps 10000 --tables-per-step 8 --eval-every 250 --save-every 250 \
  --device cuda --threads 4 \
  --real-data-dir /home/space/datasets/tabpfn_val
```

`--steps` is the **total** target step, not the number of extra steps. Model/prior
configuration, optimizer state (including LR), validation tables, and Python,
NumPy and Torch RNG state are restored from the checkpoint. Keep the accumulation
count unchanged. Real validation options must be supplied again. Reproducibility
is checked on CPU; bitwise identity across different GPU hardware is not promised.
Checkpoint files use Python serialization, so load only trusted checkpoints.

## 5. Compare baselines on independent tasks

```bash
uv run --no-sync python -m tabicl_reproduction.evaluation \
  --checkpoint outputs/ms1_run/best.pt --dataset prior \
  --tasks 100 --seed 2027 --device cuda --output outputs/ms1_prior_evaluation

uv run --no-sync python -m tabicl_reproduction.evaluation \
  --checkpoint outputs/ms1_run/best.pt --dataset tabpfn_val \
  --data-dir /home/space/datasets/tabpfn_val \
  --device cuda --output outputs/ms1_real_evaluation

uv run --no-sync python -m tabicl_reproduction.plot_training --run outputs/ms1_run
```

Baselines: class-prior/majority predictor, standardized logistic regression,
and random forest. All use the identical table split as the neural network.
Scores are computed per table and macro-averaged (not pooled over rows).
`common_*` summary fields restrict comparisons to tables where **every** method
succeeded. The per-table CSV preserves failures and timings. Inference time
includes fit/context setup and predictions, not foundation-model pretraining.

The data loader encodes test classes unseen in training as -1. They count as
incorrect in accuracy. `log_loss_known` and `balanced_accuracy_known` apply only
to test classes observed in training and are accompanied by `known_fraction`.
They must not be described as unconditional full-test scores when coverage <1.

Do not tune on the final held-out synthetic seed. The supplied real datasets are
development/validation datasets, not an independent final benchmark.

## 6. Use the classifier

```python
from tabicl_reproduction.classifier import Classifier
from tabicl_reproduction.experiment import load_checkpoint

model, checkpoint = load_checkpoint("outputs/ms1_run/best.pt", device="cpu")
classifier = Classifier(model, device="cpu", batch_size=256)
classifier.fit(X_train, y_train)  # Stores context; never updates neural weights.
probabilities = classifier.predict_proba(X_test)
predictions = classifier.predict(X_test)
```

`X` must be a numeric 2D array or tensor. String/categorical labels are supported;
categorical features must be encoded first. NaN imputation is learned on training
rows only; all-missing training columns receive zero. The model itself scales
features from training rows only. The OpenML reader now fits categorical
vocabularies after splitting, using training rows only. Unseen test categories
are treated as missing. Integer encoding plus mean imputation is a simple
prototype choice, not a claim of optimal categorical preprocessing.

## 7. Data figures

```python
import matplotlib.pyplot as plt
from tabicl_reproduction.data import load_data
from tabicl_reproduction.viz import show

loader = iter(load_data("prior", n_train=64, n_test=32, max_features=4))
samples = [next(loader) for _ in range(4)]
fig = show(samples, outfile="outputs/prior_examples.png")
plt.close(fig)
```

## Quality checks

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync ty check
uv run --no-sync pytest
```

Tests include actual backward/optimization, overfitting a single diagnostic task,
classification label decoding, context-only fit, test chunking equivalence,
validation-before-training cache safety, checkpoint/RNG round trips, and exact
CPU interrupted-versus-uninterrupted training. Overfitting a diagnostic task is
not evidence of generalization. Existing network tests remain opt-in.

## Before submission

- Train until the measured curves support convergence; evaluate independent prior
  tasks and compare against baselines. Do not claim success from a smoke run.
- Verify the OpenML registry against the assignment source and report coverage.
- Run on your actual cluster; allocation/partition/account details are local to
  the university and are not guessed here.
- Include evidence and limitations in the <=10-page report, not source code.
- Review changes as a team, then tag the agreed submission commit `ms1`.
