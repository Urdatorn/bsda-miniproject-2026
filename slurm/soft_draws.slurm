#!/bin/bash
#SBATCH --job-name=bsda-soft-draws
#SBATCH --account=naiss2026-3-353-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=04:30:00
#SBATCH --array=0-23
#SBATCH --output=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%A_%a.out
#SBATCH --error=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026/.logs/%x-%A_%a.err

# Stage 2 of the soft model, once per state of the LDA's Gibbs chain and four
# times under the mean phi: every document's posterior by NUTS (CmdStan, one
# process per core; the GPU is only the price of admission to the partition).
# Stage 1 (results/soft/lda/phi_draws.npz, 20 states) is fitted beforehand with
#   python scripts/soft_model.py --data data --lda results/soft/lda --fit-lda-only
# Tasks 0-19: phi and alpha of state k      -> results/soft/draws/dKK   (pooled by scripts/soft_pool.py)
# Tasks 20-23: the mean phi under the LDA's alpha, Dirichlet(1), (0.1), (2)
#                                           -> results/soft/mean_alpha_{lda,1,0.1,2}  (the prior check)
# Every task writes profiles.csv, comparisons.csv, diagnostics.csv.gz,
# check_documents.json and set_draws.npz; no theta_draws.npz.
#   sbatch slurm/soft_draws.slurm
#   sbatch --array=20-23 slurm/soft_draws.slurm      # the prior check alone
#   SAMPLES=500 sbatch --array=0 slurm/soft_draws.slurm
set -Eeuo pipefail
trap 'status=$?; echo "[$(date --iso-8601=seconds)] FATAL line=${LINENO} command=${BASH_COMMAND} status=${status}" >&2; exit "$status"' ERR
repo=/nobackup/proj/flash/dionysus/personal/cleland/bsda-miniproject-2026
python=/nobackup/proj/flash/dionysus/personal/cleland/sphragis_models/.venv-greberta/bin/python
export CMDSTAN=/nobackup/proj/flash/dionysus/personal/cleland/stan-aarch64/cmdstan-2.37.0
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$repo"; mkdir -p .logs
module purge; module load GPU/buildenv-gcc/2025b-eb
k=$SLURM_ARRAY_TASK_ID
case $k in
  2[0-3]) alphas=(lda 1 0.1 2); alpha=${alphas[$((k - 20))]}; phi=mean; out=results/soft/mean_alpha_$alpha ;;
  *)      alpha=lda; phi=$k; out=$(printf "results/soft/draws/d%02d" "$k") ;;
esac
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} task=$k revision=$(git rev-parse HEAD) cpus=${SLURM_CPUS_PER_TASK} phi=$phi alpha=$alpha out=$out samples=${SAMPLES:-1000} args=$*"
file "$CMDSTAN/bin/stanc" | grep -q aarch64 || { echo "stanc in $CMDSTAN is not aarch64" >&2; exit 1; }
[ -f results/soft/lda/phi_draws.npz ] || { echo "results/soft/lda/phi_draws.npz is missing: fit stage 1 first" >&2; exit 1; }
# a binary compiled on the login node (x86, by a dry run) must not be reused here; each task compiles its own copy
[ -f lda_given_phi ] && ! file lda_given_phi | grep -q aarch64 && rm -f lda_given_phi lda_given_phi.hpp
stan_copy=$(mktemp -d)/lda_given_phi.stan; cp lda_given_phi.stan "$stan_copy"
srun --kill-on-bad-exit=1 "$python" -u scripts/soft_model.py --data data --lda results/soft/lda --stan "$stan_copy" \
    --phi "$phi" --alpha "$alpha" --out "$out" --samples "${SAMPLES:-1000}" --no-draws "$@"
rm -rf "$(dirname "$stan_copy")"
echo "[$(date --iso-8601=seconds)] completed task $k -> $out"
