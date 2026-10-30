#!/bin/bash
#SBATCH --job-name=bsda-soft-trajectory
#SBATCH --account=naiss2026-3-353-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=01:15:00
#SBATCH --output=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%j.out
#SBATCH --error=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%j.err

# The soft model along the walk: the posterior mean profile of every fifth
# recorded state of every chain (about 4,700 documents with the topic round), under the mean phi and the
# LDA's alpha, by NUTS (CmdStan, one process per core), into
# results/soft/trajectory/trajectory.csv, which the report's section on the
# lift along the walk and the paper's figure read. Stage 1 must be in
# results/soft/lda (slurm/soft_draws.slurm's prerequisite).
#   sbatch slurm/soft_trajectory.slurm
#   sbatch slurm/soft_trajectory.slurm --every 1      # every recorded state
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FATAL line=${LINENO} command=${BASH_COMMAND} status=${status}" >&2; exit "$status"' ERR
repo=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026
python=/nobackup/proj/flash/dionysus/personal/cleland/sphragis_models/.venv-greberta/bin/python
export CMDSTAN=/nobackup/proj/flash/dionysus/personal/cleland/stan-aarch64/cmdstan-2.37.0
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$repo"; mkdir -p .logs
module purge; module load GPU/buildenv-gcc/2025b-eb
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} revision=$(git rev-parse HEAD) cpus=${SLURM_CPUS_PER_TASK} args=$*"
file "$CMDSTAN/bin/stanc" | grep -q aarch64 || { echo "stanc in $CMDSTAN is not aarch64" >&2; exit 1; }
[ -f results/soft/lda/phi_draws.npz ] || { echo "results/soft/lda/phi_draws.npz is missing: fit stage 1 first" >&2; exit 1; }
# a binary compiled on the login node (x86, by a dry run) must not be reused here
[ -f lda_given_phi ] && ! file lda_given_phi | grep -q aarch64 && rm -f lda_given_phi lda_given_phi.hpp
srun --kill-on-bad-exit=1 "$python" -u scripts/soft_trajectory.py --data data --lda results/soft/lda --out results/soft/trajectory "$@"
echo "[$(date --iso-8601=seconds)] completed"
