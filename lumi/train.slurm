#!/usr/bin/env bash
# SLURM script: train noisy-diffqec on a single LUMI-G node (one MI250X GCD).
#
# Submit with:
#     sbatch lumi/train.slurm           # uses the default account below
#     sbatch --account=project_xxx lumi/train.slurm   # override per-job
#
# The script defaults to a moderate d=5 run that fits in <2 GB of VRAM and
# finishes in well under the 30-minute queue limit.  Override any of the
# export variables below (or use `sbatch --export=...`) to scale up.
#
# Environment variables understood by this script:
#     DISTANCE      code distance                (default 5)
#     ROUNDS        stabilizer rounds             (default 5)
#     SHOTS         training+eval shots per cell  (default 20000)
#     EPOCHS        training epochs               (default 30)
#     HIDDEN        model hidden size             (default 96)
#     NOISE         emerald | garnet | phenomenological (default emerald)
#     T             diffusion steps               (default 100)
#     PARTITION     LUMI partition                (default small-g)
#     TIME_LIMIT    SLURM time limit              (default 00:30:00)
#     ACCOUNT       SLURM billing project         (override on cmdline)

#SBATCH --job-name=noisy-diffqec
#SBATCH --account=project_465003017
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=lumi/logs/train-%j.out
#SBATCH --error=lumi/logs/train-%j.err

set -euo pipefail

PROJECT_ROOT="/scratch/project_465003017/diffqec-noisy"
VENV_DIR="${PROJECT_ROOT}/.venv-lumi"
CKPT_DIR="${PROJECT_ROOT}/ckpts/lumi-train-${SLURM_JOB_ID:-manual}"

mkdir -p "${PROJECT_ROOT}/lumi/logs" "${CKPT_DIR}"

echo "[train.slurm] job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "[train.slurm] project=${PROJECT_ROOT}  ckpt=${CKPT_DIR}"

# Load the currently-supported LUMI stack (LUMI/24.03 is deprecated).
module load LUMI/25.03 2>/dev/null || module load LUMI/24.11 2>/dev/null || true
module load partition/G 2>/dev/null || true
source "${VENV_DIR}/bin/activate"

export HF_HOME=/scratch/${SLURM_JOB_USER:-${LUMI_USER:-siljheis}}/hf-cache
export TORCH_HOME=/scratch/${SLURM_JOB_USER:-${LUMI_USER:-siljheis}}/torch-cache
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

cd "${PROJECT_ROOT}"

python -m scripts.train_noisy_diffqec \
    --distance  "${DISTANCE:-5}" \
    --rounds    "${ROUNDS:-5}" \
    --shots     "${SHOTS:-20000}" \
    --epochs    "${EPOCHS:-30}" \
    --hidden    "${HIDDEN:-96}" \
    --num-denoiser-layers 4 \
    --noise     "${NOISE:-emerald}" \
    --T         "${T:-100}" \
    --scale     1.0 \
    --device    cuda \
    --seed      42 \
    --checkpoint-dir "${CKPT_DIR}"

echo "[train.slurm] finished.  Checkpoints at ${CKPT_DIR}"
