# Report

```
report/
├── report_template.tex   main file: preamble + one \input per section
├── report_template.bib   references (BibTeX)
├── macros.tex            shared names (\tabicl{}, \tabpfnval{}) and \todo{}
├── sections/             one file per section of the task sheet (see below)
├── figures/              generated figures (PDF), committed
└── plots/                scripts that make the figures
```

The sections follow the report part of the Milestone 1 task sheet; each file
lists that section's guiding questions as comments.

| File | Section | Points |
|---|---|---|
| `01_introduction.tex` | Introduction (short) | – |
| `02_dataset_overview.tex` | Dataset overview | 4 |
| `03_baseline_evaluation.tex` | Baseline and evaluation | 3 |
| `04_discussion.tex` | Discussion | 2 |

Rules from the task sheet: at most 10 pages, no code in the report, submit
the PDF on ISIS, and tag the submitted commit `ms1`.

## Build

In VS Code (LaTeX Workshop), open any `.tex` file and save: the PDF rebuilds
and opens next to it. Section files start with `% !TEX root =
../report_template.tex`, so saving a section builds the whole report.
Double-click in the PDF jumps to the source line.

From a terminal: `cd report && latexmk -pdf report_template.tex`.

## Writing

- One paragraph, one idea. As much as necessary, as little as possible.
- Labels: `sec:`, `fig:`, `tab:` + a short name. Reference with `\Cref{fig:...}`
  (writes "Figure 1"), so numbers never go stale.
- Mark open points with `\todo{...}` (red in the PDF); remove all before
  handing in.
- Cite with `\citet{key}` in running text ("Author (2025) show ...") and
  `\citep{key}` otherwise ("... (Author, 2025)"). Add the BibTeX entry to
  `report_template.bib`; Google Scholar and the paper's arXiv page export it.

## Figures

Every figure is made by a script in `plots/`, never by hand, so it can be
remade when the results change. One command makes all of them from the
training run and the results CSVs (`results/*.csv` by default):

```bash
uv run python report/plots/make_figures.py outputs/runs/<date>/<time>
```

| Figure | Shows |
|---|---|
| `training_curves` | training loss (prior) and `tabpfn_val` log loss over the steps; overfitting check |
| `roc_auc_over_training` | our mean ROC AUC on `tabpfn_val` over the steps, baselines as lines |
| `overall_roc_auc` | mean ROC AUC per model with 95% CIs, prior vs. OpenML tables |
| `mean_rank` | mean rank per model (1 = best per table), prior vs. OpenML tables |
| `per_dataset` | our ROC AUC vs. the best baseline's, one point per table |
| `roc_auc_by_category` | mean ROC AUC by number of classes, features and training rows |
| `confusion_matrices` | ours vs. the best baseline on the OpenML table we win most / lose most |

The Section 2 figures need the data, not a run, so they have their own command:

```bash
uv run python -m tabicl_reproduction.explore   # report/data_stats.csv
uv run python report/plots/data_overview.py
```

| Figure | Shows |
|---|---|
| `example_tables` | `show()` on the first 3 saved prior tables and 3 OpenML tables |
| `table_shapes` | share of tables per training rows, features and classes, prior vs. OpenML |

The plot code is in `plots/training_curves.py` (run logs), `plots/comparison.py`
(results CSVs) and `plots/confusion.py`; every function takes the data and an
`outfile`. They share `plots/style.py`: same fonts, colors (one per model) and
width, saved as vector PDF into `figures/`. To add a figure, write a function
next to the others, call `save(fig, outfile)`, add it to `make_figures.py` and
a test in `tests/report/`. Then in a section:

```latex
\begin{figure}
  \centering
  \includegraphics[width=\linewidth]{<name>}
  \caption{What is shown: axes, data, run. No interpretation here.}
  \label{fig:<name>}
\end{figure}
```

Checklist (from the course's report guidelines):

- Caption describes what is shown; the interpretation is in the text.
- Referenced and explained in the text: what can be seen, does it make sense?
- Axis labels with units, a legend when there is more than one line.
- Same axis limits for figures that are compared.
- Readable colors (the shared palette is colorblind-safe), vector PDF.
