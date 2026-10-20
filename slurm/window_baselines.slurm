#!/bin/bash
#SBATCH --job-name=bsda-baselines
#SBATCH --account=naiss2026-3-353-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%j.out
#SBATCH --error=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%j.err

# The baseline windows (Greek at large from the OGA parses, and the tragedians).  Compute nodes have no
# network: download the shards on the login node first,
#   HF_HOME=<cache> hf download Urdatorn/oga-conllu-stoicheia --repo-type dataset
# and pass the snapshot's data directory:  sbatch slurm/window_baselines.slurm --oga <dir>
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FATAL line=${LINENO} command=${BASH_COMMAND} status=${status}" >&2; exit "$status"' ERR
repo=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026
python=/nobackup/proj/flash/dionysus/personal/cleland/sphragis_models/.venv-greberta/bin/python
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$repo"; mkdir -p .logs
module purge; module load GPU/Python/3.13.5-bare-gcc-2025b-eb
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} revision=$(git rev-parse HEAD) args=$*"
srun --kill-on-bad-exit=1 "$python" -u scripts/window_baselines.py --data data --out data "$@"
echo "[$(date --iso-8601=seconds)] completed"
