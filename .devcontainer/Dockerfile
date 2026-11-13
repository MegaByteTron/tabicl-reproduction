FROM python:3.13-slim AS linux-base

# Utilities (git is needed for the commit hash stored in each run's config)
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends build-essential \
    sudo curl git htop less rsync screen vim nano wget ca-certificates openssh-client zsh procps psmisc \
    jq gfortran pkg-config

# Slurm: wrappers that run the real commands on the login node over ssh, so
# submitit can call sbatch from inside the container
RUN COMMANDS="sacct sacctmgr salloc sattach sbatch sbcast scancel scontrol sdiag sgather sinfo smap sprio squeue sreport srun sshare sstat strigger sview" \
    && for CMD in $COMMANDS; do echo '#!/bin/bash' > "/usr/local/bin/$CMD" \
    && echo 'ssh $USER@$SLURM_CLUSTER_NAME -t "cd $PWD; . ~/.zshrc 2>/dev/null || . ~/.bashrc 2>/dev/null; bash -lc '\'$CMD \$@\''"' >> "/usr/local/bin/$CMD" \
    && chmod +x "/usr/local/bin/$CMD"; done

# Non-root user
RUN useradd -m -s /bin/bash devuser

FROM linux-base AS python-base

WORKDIR /srv/repo

# Environment variables
ENV UV_PROJECT_ENVIRONMENT="/venv"
ENV UV_PYTHON_INSTALL_DIR="/python"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=python3.13
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
# Only the dependencies are installed in the image, not our package (it lives in
# src/ and is edited on the cluster). A relative "src" resolves against the
# working directory, which Apptainer sets to the checkout the job is started in,
# so the checkout's code is always the one that runs. "/srv/repo/src" covers the
# JOB_REPO bind mount (-B $JOB_REPO:/srv/repo --pwd /srv/repo) set by the template.
ENV PYTHONPATH="src:/srv/repo/src:$PYTHONPATH"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
RUN mkdir -p /venv && chown devuser:devuser /venv

USER devuser

# Default groups only: runtime dependencies plus dev (pytest, ruff, ty), so the
# tests can also run in the container. The notebook group is left out.
RUN --mount=type=cache,target=/home/devuser/.cache/uv,uid=1000 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project
