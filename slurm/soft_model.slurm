#!/bin/bash
#SBATCH --job-name=bsda-soft
#SBATCH --account=naiss2026-3-353-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%j.out
#SBATCH --error=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%j.err

# The soft model in one go: stage 1 (the seeded LDA for phi and alpha, unless
# results/soft/lda already holds it) and stage 2 under the mean phi and the
# LDA's alpha, every document's posterior by NUTS (CmdStan, one process per
# core), into results/soft/run.  The GPU is only the price of admission to the
# partition; the work is on the Grace cores.  The report's results come from
# slurm/soft_draws.slurm (one stage-2 run per state of phi, pooled by
# scripts/soft_pool.py); this script is for a single run with other settings.
# The dataset must be in data/ (`hf download Urdatorn/bsda-miniproject-2026
# --repo-type dataset --local-dir data`).
#   sbatch slurm/soft_model.slurm
#   sbatch slurm/soft_model.slurm --alpha 1 --out results/soft/run_flat
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FATAL line=${LINENO} command=${BASH_COMMAND} status=${status}" >&2; exit "$status"' ERR
repo=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026
python=/nobackup/proj/flash/dionysus/personal/cleland/sphragis_models/.venv-greberta/bin/python
export CMDSTAN=/nobackup/proj/flash/dionysus/personal/cleland/stan-aarch64/cmdstan-2.37.0
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$repo"; mkdir -p .logs
module purge; module load GPU/buildenv-gcc/2025b-eb
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} revision=$(git rev-parse HEAD) cpus=${SLURM_CPUS_PER_TASK} $(g++ --version | head -1) args=$*"
file "$CMDSTAN/bin/stanc" | grep -q aarch64 || { echo "stanc in $CMDSTAN is not aarch64" >&2; exit 1; }
# a binary compiled on the login node (x86, by a dry run) must not be reused here
[ -f lda_given_phi ] && ! file lda_given_phi | grep -q aarch64 && rm -f lda_given_phi lda_given_phi.hpp
srun --kill-on-bad-exit=1 "$python" -u scripts/soft_model.py "$@"
echo "[$(date --iso-8601=seconds)] completed"
