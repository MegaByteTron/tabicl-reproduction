# NanoTabICL: a minimal TabICLv2 implementation

| [Full TabICLv2 code](https://github.com/soda-inria/tabicl) | [TabICLv2 Paper](https://arxiv.org/abs/2602.11139) |
|----------------------------------------------------|------------------------|

This repository provides a short implementation of the [TabICLv2](https://arxiv.org/abs/2602.11139) architecture (<170 LOC)
and a slightly simplified implementation of the TabICLv2 prior (<330 LOC).
For using our pre-trained TabICLv2 model, 
please visit the [main repository](https://github.com/soda-inria/tabicl). (Pre-training code coming out soon.)

Compared to [nanoTabPFN](https://github.com/automl/nanoTabPFN),

- the model is TabICLv2, not TabPFN,
- we implement the full model (including RoPE, QASSMax, etc.) with speed optimizations
  (but without the inference wrappers + memory optimizations from the main repository),
- we implement regression as well,
- we provide a (slightly simplified but still well-performing) prior for dataset generation,
- we currently do not provide a sklearn interface,
- we currently do not provide pre-training code.
  For now, we refer to [nanoTabPFN](https://github.com/automl/nanoTabPFN) and
  the [nanoTabPFN speedrun](https://github.com/borawhocodess/modded-nanotabpfn) for pre-training code.
  
Note that this repo uses LayerNorm with bias, which is used by the classification checkpoint of TabICLv2,
while the regression checkpoint of TabICLv2 uses LayerNorm without bias.

## Model usage

```python
from model import NanoTabICLv2
import torch

model = NanoTabICLv2(max_classes=10, out_dim=10)  # original model size
X_train_and_test = torch.randn(batch_size, n_train + n_test, n_cols)
y_train = torch.randint(10, size=(batch_size, n_train)).float()
y_test_pred_logits = model(X_train_and_test, y_train)

# if you want a smaller model + regression with 999 quantiles instead
# warning: for regression, you need to standardize y yourself (and backtransform the output)
model = NanoTabICLv2(
    max_classes=0,
    out_dim=999,
    embed_dim=96,
    col_num_blocks=2,
    row_num_blocks=2,
    icl_num_blocks=4,
    col_nhead=4,
    row_nhead=4,
    icl_nhead=4,
)
y_train = torch.randn(batch_size, n_train)
y_test_pred_quantiles = model(X_train_and_test, y_train)
```

Note that `X_train_and_test` is standardized inside the model (based on train only). 
We do not include other preprocessing options from TabICLv2 
since they are normally not part of the architecture.

## Nanoprior

In `prior.py`, we provide a concise and slightly simplified 
implementation of the TabICLv2 prior,
which nevertheless performs similarly well in our 
(smaller-scale) experiments. 
Changes compared to the TabICLv2 prior are:

- No correlated sampling of scalar variables / hyperparameters.
- Removed graph filtering (should be captured by dataset filtering anyway).
- No graph pruning of irrelevant nodes 
(can be a bit slower, but yields the same output).
- Categorical converters with cardinality k always use k dimensions,
never fewer.
- Kumaraswamy warping now affects the extracted dataset column, 
not the propagated value.
- Fixed the constant in the random EM function.
- The random activation used for random activation matrices 
follows the general random activation now 
by only using two of the four possible random power activation types.
- Use corrected categorical size limit of 100 instead of 10.
- Dataset filtering uses the same categorical sizes 
in every attempt until a non-filtered dataset is generated.
- fill NaN/inf with zero instead of discarding the whole dataset.
- (Use torch.sign() in random activation,
which has a different behavior at 0 than `2*(x>=0).float()-1`.)

Preprocessing code (outlier handling, standard scaling) 
is also not included as it could be done inside the model.

When running the file, it will generate a plot of some random datasets.

## Our model as a scikit-learn classifier

`TabICLClassifier` wraps a pretrained model in the scikit-learn interface.
`fit` only stores the training rows (in-context learning, no weights
change); `predict_proba` runs the forward pass, averaged over 8 column
orders because the model depends on the order of the columns.

```python
from tabicl_reproduction.model import TabICLClassifier

clf = TabICLClassifier.from_checkpoint("outputs/runs/<date>/<time>")  # best.pt
clf.fit(X_train, y_train)          # any labels; clf.classes_ keeps them
proba = clf.predict_proba(X_test)  # column k is clf.classes_[k]
pred = clf.predict(X_test)
```

It works wherever a scikit-learn classifier does, e.g. `DecisionBoundary` in
`show()`. Tables with more classes than the model supports (10) raise
`UnsupportedTableError`.

## Evaluation: our model vs. the baselines

Every model is scored by the same code (`tabicl_reproduction/evaluation/`):

- `score_tables(estimator, tables)` fits any estimator on each table's
  training rows, predicts the test rows and computes accuracy, balanced
  accuracy, log loss, ROC AUC and ECE. It writes nothing; it returns the rows.
- Pretraining validation (`evaluator=tabpfn_val`) calls it with our model
  (one column order, to stay fast) to log `eval/*` and pick `best.pt`.
- The experiment calls it for every model and saves the results.

The baselines are k-NN, logistic regression, random forest and
gradient-boosted trees (scikit-learn defaults), fitted on every table.

```bash
uv run python -m tabicl_reproduction.data.prior_val          # once: 1,600 prior tables
uv run python scripts/compare.py --limit 5                   # quick test
uv run python scripts/compare.py                             # all baselines
uv run python scripts/compare.py --checkpoint <run dir>      # + our best.pt
uv run python scripts/compare.py --checkpoint <run dir> --models tabicl_best
uv run python scripts/compare.py --summary                   # summary only
```

Tables: every `tabpfn_val` dataset with 5-fold stratified cross-validation
(at most 2,048 training rows per fold; `--folds` changes the number), and the
saved `prior_val` tables with their fixed split. Outputs, in `results/`:

| File | Contents |
|---|---|
| `<model>.csv` | One row per table and fold: table size, the five metrics, fit and predict time, `ok` or `skipped` with the reason, checkpoint step, git commit. |
| `summary.txt` | Per model and source: mean, std and mean rank of each metric. Folds are averaged per table first, and only tables every model scored are counted. |
| `predictions/<model>/<source>/<table>_split<k>.npz` | `y_true`, `proba` and `class_names` of every test row, for confusion matrices and ROC curves (not committed). |

```python
from tabicl_reproduction.evaluation.results import load_predictions, load_results

df = load_results("results/*.csv")           # all models in one DataFrame
p = load_predictions("results/predictions/tabicl_best/tabpfn_val/31_split0.npz")
# p["proba"][:, k] is class p["class_names"][k]; y_true -1: class unseen in training
```

Scripts writing to a model's CSV refuse to append to an existing one; move it
first. Run `compare.py` from the repo root (locally, or in the container shell
on the cluster), not as a SLURM job: a job runs in a temporary copy of the
repo, so `results/` would be lost. On a GPU, add `--device cuda`.

`load_data("prior_val")` reads the saved prior tables
(`.cache/datasets/prior_val` locally, `/home/space/datasets/prior_val` on the
cluster). To share them, copy the folder to the cluster and make it readable:

```bash
scp -r .cache/datasets/prior_val <user>@<cluster>:/home/space/datasets/
ssh <user>@<cluster> chmod -R a+rX /home/space/datasets/prior_val
```

## Updates

- 2026/08/17: Replace class-embedding with one-hot + linear, which has a bias compared to the nn.Embedding used before, and matches the full model implementation.
- 2026/06/10: Add nanoprior. 
Bugfix based on [#2](https://github.com/soda-inria/nanotabicl/issues/2): Subtract mean when standardizing input data.
- 2026/03/25: Add faster + cached RoPE implementation based on the TabICLv2 version (warning: this permutes the neurons, so it's not compatible with older nanotabicl checkpoints).
