#!/usr/bin/env bash
# SLURM script: train noisy-diffqec on a single LUMI-G node (one MI250X GCD).
#
# Submit with:
#     sbatch --account=<your-project> lumi/train.slurm
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
#SBATCH --partition=${PARTITION:-small-g}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=${TIME_LIMIT:-00:30:00}
#SBATCH --output=lumi/logs/train-%j.out
#SBATCH --error=lumi/logs/train-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-lumi}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/ckpts/lumi-train-${SLURM_JOB_ID:-manual}}"

mkdir -p lumi/logs "${CKPT_DIR}"

echo "[train.slurm] job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "[train.slurm] project=${PROJECT_ROOT}  ckpt=${CKPT_DIR}"

module load LUMI/24.03 partition/G rocm/6.0.3 2>/dev/null || true
source "${VENV_DIR}/bin/activate"

export HF_HOME=/scratch/${SLURM_JOB_USER:-$(id -un)}/hf-cache
export TORCH_HOME=/scratch/${SLURM_JOB_USER:-$(id -un)}/torch-cache
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
