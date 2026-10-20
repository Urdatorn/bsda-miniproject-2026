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

# The soft model: seeded LDA for phi, then every document's posterior by NUTS
# (CmdStan, one process per core).  The GPU is only the price of admission to
# the partition; the work is on the Grace cores.  Compute nodes have no
# network: run `hf download Urdatorn/bsda-miniproject-2026 --repo-type dataset
# --local-dir data` first.
#   sbatch slurm/soft_model.slurm
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
srun --kill-on-bad-exit=1 "$python" -u scripts/soft_model.py "$@"
echo "[$(date --iso-8601=seconds)] completed"
