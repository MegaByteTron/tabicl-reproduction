# Running on the cluster

A practical guide for people who have never used SLURM. Everything below is typed
by you, on the cluster, by hand.

**Never use coding agents (Claude Code, Copilot agents, etc.) on the cluster.
Agents on the TU Berlin cluster get the account permanently banned. Do not
install them there and do not point them at it from your laptop.**

## The basics

- **SLURM** is the queue of the cluster. You do not run heavy code on the login
  node; you ask SLURM for a machine (a "job") and it starts your code when one is
  free.
- **Partitions** are queues with a fixed time limit. A job is **killed** when it
  reaches the limit of its partition. Shorter partitions start sooner.
- **Apptainer** runs our container image (Python, PyTorch, all dependencies).
  The image holds only the libraries. Our code is read from your `git clone`.
- **submitit** (via the project template) turns
  `python ... cfg/job=gpu_2h` into a SLURM submission, so no `.sh` files are needed.

Run locally first: see
[`training/README.md`](../src/tabicl_reproduction/training/README.md). Add
`cfg/job=<preset>` to any of those commands to submit the same run to the cluster.

## Partitions and presets

| Preset (`cfg/job=`) | Partition | Limit | GPU | CPUs |
|---|---|---|---|---|
| `test` | gpu-teaching-9m | 9 min | any | 4 |
| `gpu_2h` | gpu-teaching-2h | 2 h | 40/80 GB, H100, H200 | 8 |
| `gpu_5h` | gpu-teaching-5h | 5 h | 40/80 GB, H100, H200 | 8 |
| `sweep` | gpu-teaching-2h | 2 h per worker | 40/80 GB, H100, H200 | 8 |

Students may only use the GPU-TEACHING partitions (2D, 2H, 5H, 7D, 9M); never
submit to the other gpu-* partitions. There is no preset for gpu-teaching-2d
and gpu-teaching-7d on purpose:
queue times are long and a crash loses days.

How to choose:

- Pick the shortest partition that fits. Short jobs start sooner.
- Always try new code on `test` first.
- Long training is a **chain** of `gpu-teaching-2h` / `gpu-teaching-5h` jobs. Each job saves a
  checkpoint before its time runs out; start the next one with
  `resume=<old run dir>` (the `outputs/...` folder printed when the first job was
  submitted).

## One-time setup

1. **Weights & Biases.** Create an account at wandb.ai, create a project, copy your
   API key.
2. **Clone the repo** into your home on the cluster and create `.env` in it:
   ```bash
   git clone https://github.com/MegaByteTron/tabicl-reproduction.git
   cd tabicl-reproduction
   cat > .env <<'EOF'
   WANDB_API_KEY=<your key>
   WANDB_ENTITY=<your wandb user or team>
   WANDB_PROJECT=<project name>
   EOF
   ```
   `.env` is git-ignored. Never commit it.
3. **Registry login.** Create a GitHub token at github.com/settings/tokens with
   `read:packages` and `write:packages`, then on the cluster:
   ```bash
   apptainer remote login --username <github user> docker://ghcr.io
   ```
   and paste the token as the password.
4. **Passwordless ssh to the login node** (submitit calls `sbatch` from inside the
   container through ssh). On the cluster:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
   ssh $USER@$SLURM_CLUSTER_NAME exit    # must return without any prompt
   ```
5. **Build the image once** (on your laptop, in the repo): push a tag so GitHub
   Actions builds and publishes it.
   ```bash
   git tag v0.0.1 && git push --tags
   ```
   Wait until the workflow in the repo's Actions tab is green. Make the package
   visible to the cluster (repo page -> Packages -> `tabicl-reproduction` ->
   settings) or keep it private and use the login from step 3. Push a new tag
   whenever `pyproject.toml` / `uv.lock` change.

## Starting

Get a shell inside the container (on a CPU node, no GPU needed for editing and
checks):

```bash
srun --partition=cpu-2h --pty bash
apptainer run --writable-tmpfs -B /home/space/datasets oras://ghcr.io/megabytetron/tabicl-reproduction:latest-sif /bin/bash
cd tabicl-reproduction
```

The first run downloads the image and can take a few minutes. `-B` makes the
cluster's shared datasets folder visible in the container; apptainer only
mounts your home by default. Submitted jobs mount it on their own.

### Benchmark data

The OpenML datasets live in `/home/space/datasets/<dataset>`, readable by
everyone (the code runs the equivalent of `chmod -R a+rX` on each dataset it
downloads). Download them once from the container shell before the first run
with an evaluator, so jobs don't spend their time limit downloading:

```bash
python -c "
from tabicl_reproduction.data.loader import default_data_dir
from tabicl_reproduction.data.openml_tables import OpenMLDataset
from tabicl_reproduction.data.benchmarks import BENCHMARKS
d = OpenMLDataset(datasets=BENCHMARKS['tabpfn_val'], data_dir=default_data_dir())
print(d.data_dir, 'failures:', d.download_all())
"
```

Datasets already there are skipped. From this shell submit a job:

```bash
python scripts/train.py model=small cfg/job=gpu_2h
```

Run commands from the repo root. The image finds our code through a relative
`PYTHONPATH=src`. Evaluation works the same way:
`python scripts/evaluate.py model=small checkpoint=<run dir> cfg/job=gpu_2h`.

Without the container shell, `cluster/debug.sbatch` is a plain `sbatch` fallback
(`mkdir -p logs && sbatch cluster/debug.sbatch model=small`).

### One command: `bin/slurm-agent`

Instead of the two steps above, run this from the repo root on the cluster:

```bash
bin/slurm-agent scripts/train.py model=small cfg/job=gpu_2h
```

It starts a short allocation on `gpu-teaching-9m`, opens the container there and
submits the job. It only runs **committed and pushed** code: each commit gets
its own checkout under `.worktrees/<sha>` (delete old ones with
`rm -rf .worktrees/<sha> && git worktree prune`), so you can keep editing while
jobs run. Run outputs go to `outputs/<timestamp>-<sha>`. Settings via
environment variables: `REF=<branch|sha>`, `IMAGE=oras://...:<tag>-sif`,
`CPU_PART=<partition>`. From your laptop, `HOST=<cluster> bin/slurm-agent ...`
does the same over ssh (the checkout must be at the same path under `$HOME`).

Despite its name, this is a plain shell script: type it yourself. Never let a
coding agent run it.

## Test order

Do these in order; do not skip ahead.

1. **Import check** (finishes in about a minute):

   ```bash
   python scripts/train.py cfg/job=test \
     model=tiny '~evaluator' epochs=1 steps_per_epoch=2
   ```

   `~evaluator` skips the benchmark evaluation. The log must show the config,
   `"device": "cuda"` in the run folder's `run.json`, and no import errors.
2. **Time-limit test**: start a run that is too long for the 9 minutes of
   `gpu-teaching-9m`, so SLURM has to stop it:

   ```bash
   python scripts/train.py cfg/job=test \
     model=small '~evaluator' epochs=1000 steps_per_epoch=20 checkpoint_every_min=2
   ```

   SLURM sends SIGUSR2 to submitit's process, which passes it on to the
   training and waits until it has saved. Near the end, the job's
   `<jobid>_0_log.out` (in the submission folder) must show, in this order,
   `Time limit: asking the training to stop and save.`,
   `Received SIGUSR2, stopping after this step.` and `Stopped at step ...`.
   `<jobid>_0_log.err` then says `Job not requeued because: checkpoint function
   returned None.` That is expected: the job ends after the save and you resume
   by hand. In the training run folder (`outputs/runs/...`), `run.json` must
   have `"status": "stopped"` and `"signal": "SIGUSR2"`, the last
   `metrics.jsonl` row `"event": "stopped"`, and `last.pt` must exist. If these
   are missing, the training was killed before it could save: `last.pt` is then
   only as recent as the last epoch or 2-minute save.
   Then continue the run with `resume=<that run folder>` and check that the log
   says `Resumed from ... at step ...`.
3. **Short sweep** with `cfg/job=sweep`; check in W&B that both workers report
   `eval/log_loss`.
4. **Real runs** on `gpu_2h` / `gpu_5h`, chained with `resume=`.

## Monitoring

```bash
squeue -u $USER          # your queued and running jobs
scancel <jobid>          # cancel one job; the training saves last.pt first
                         # (scancel -u $USER cancels all)
seff <jobid>             # after it ends: CPU/memory use and exit state
```

Logs: submitit writes `<jobid>_log.out` and `<jobid>_log.err` into the run's
folder under `outputs/` (the path is printed when you submit). Hydra's config and
training outputs are in the same run folder; live curves are on W&B.

## Troubleshooting

- **Job dies after 5 minutes.** The template's default timeout is 5 minutes. All
  presets here set their own time; if you write a new preset, set `timeout_min`.
  Never set `slurm_params.time_hours`: submitit then crashes on the time-limit
  signal, and the training is killed without saving.
- **No `Stopped at step ...` at the end of the log, and `run.json` has no
  `status`.** The training was killed before it could save. Look at the end of
  `<jobid>_0_log.err` for the reason. `last.pt` is still there from the last
  epoch, so `resume=` loses at most one epoch.
- **Job ends exactly at the limit** (`seff` shows `TIMEOUT`). It was killed.
  Resume it with `resume=<old run dir>` or pick a longer partition.
- **`ModuleNotFoundError: tabicl_reproduction`.** You started from the wrong
  folder. `cd` to the repo root and use the path
  `scripts/train.py`.
- **A new dependency is missing.** The image only contains what was in
  `pyproject.toml` when the tag was built. Update, push a new tag, wait for the
  workflow, then run again (add `JOB_IMAGE=oras://...:<tag>-sif` to use a specific
  version).
- **`ssh ... Permission denied` when submitting.** Redo the passwordless ssh step.
- **No internet on the node / W&B hangs.** Use offline mode:
  `cfg.wandb.mode=offline`, then upload later with `wandb sync <run dir>`.
- **Disk.** `/home` has a 500 GB quota on shared storage. Avoid creating many small
  files (delete old `outputs/` folders and extra checkpoints).
