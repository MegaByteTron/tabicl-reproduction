#!/bin/bash
# Fallback without submitit/Python job presets. Usage, from the repo checkout:
#   mkdir -p logs && sbatch cluster/debug.sbatch model=small

# Use the gpu-teaching-9m partition (9 minute limit).
#SBATCH --partition=gpu-teaching-9m
# Ask for one GPU on the node.
#SBATCH --gpus-per-node=1
# Reserve 4 CPU cores; SLURM_CPUS_PER_TASK=4 also sets the number of prior workers.
#SBATCH --cpus-per-task=4
# Write everything the job prints to logs/<job id>.out (the logs/ folder must exist).
#SBATCH --output=logs/%j.out

# Start the container (--nv makes the GPU visible inside) and run training;
# "$@" passes along the Hydra overrides you gave to sbatch.
apptainer run --nv oras://ghcr.io/megabytetron/tabicl-reproduction:latest-sif python scripts/train.py "$@"
