#!/bin/bash
#SBATCH --job-name=bsda-round2
#SBATCH --account=naiss2026-3-353-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%j.out
#SBATCH --error=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%j.err

# Round 2 (seeded LDA + BayesFlow) on a GH200.  Compute nodes have no network: run
# `hf download Urdatorn/bsda-miniproject-2026 --repo-type dataset --local-dir data` first.
#   sbatch slurm/round2.slurm --topics clusters --out results/round2_clusters
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FATAL line=${LINENO} command=${BASH_COMMAND} status=${status}" >&2; exit "$status"' ERR
repo=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026
python=/nobackup/proj/flash/dionysus/personal/cleland/sphragis_models/.venv-greberta/bin/python
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 KERAS_BACKEND=torch
cd "$repo"; mkdir -p .logs
module purge; module load GPU/Python/3.13.5-bare-gcc-2025b-eb
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} revision=$(git rev-parse HEAD) args=$*"
srun --kill-on-bad-exit=1 "$python" -u scripts/round2_seeded_lda_bayesflow.py "$@"
echo "[$(date --iso-8601=seconds)] completed"
