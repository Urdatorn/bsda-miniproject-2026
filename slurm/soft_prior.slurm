#!/bin/bash
#SBATCH --job-name=bsda-soft-prior
#SBATCH --account=naiss2026-3-353-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%j.out
#SBATCH --error=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%j.err

# The prior check of the soft model: every document sampled again under
# Dirichlet(alpha) priors other than the report's flat Dirichlet(1), with the
# phi of the report's run (results/soft/phi_matrix.csv.gz) reused unchanged,
# so that only the prior moves.  One run per alpha in ALPHAS (default "0.1 2"),
# each into results/soft/alpha_<alpha>, about 15 minutes each on 96 cores.
#   sbatch slurm/soft_prior.slurm
#   ALPHAS="0.5 5" sbatch slurm/soft_prior.slurm
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FATAL line=${LINENO} command=${BASH_COMMAND} status=${status}" >&2; exit "$status"' ERR
repo=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026
python=/nobackup/proj/flash/dionysus/personal/cleland/sphragis_models/.venv-greberta/bin/python
export CMDSTAN=/nobackup/proj/flash/dionysus/personal/cleland/stan-aarch64/cmdstan-2.37.0
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$repo"; mkdir -p .logs
module purge; module load GPU/buildenv-gcc/2025b-eb
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} revision=$(git rev-parse HEAD) cpus=${SLURM_CPUS_PER_TASK} alphas=${ALPHAS:-0.1 2} args=$*"
file "$CMDSTAN/bin/stanc" | grep -q aarch64 || { echo "stanc in $CMDSTAN is not aarch64" >&2; exit 1; }
# a binary compiled on the login node (x86, by a dry run) must not be reused here
[ -f lda_given_phi ] && ! file lda_given_phi | grep -q aarch64 && rm -f lda_given_phi lda_given_phi.hpp
[ -f results/soft/phi_matrix.csv.gz ] || { echo "results/soft/phi_matrix.csv.gz is missing: run slurm/soft_model.slurm first" >&2; exit 1; }
for alpha in ${ALPHAS:-0.1 2}; do
  echo "[$(date --iso-8601=seconds)] prior Dirichlet($alpha) -> results/soft/alpha_$alpha"
  srun --kill-on-bad-exit=1 "$python" -u scripts/soft_model.py --data data --phi results/soft/phi_matrix.csv.gz \
      --alpha "$alpha" --out "results/soft/alpha_$alpha" --no-draws "$@"
done
echo "[$(date --iso-8601=seconds)] completed"
