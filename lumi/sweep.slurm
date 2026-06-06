#!/usr/bin/env bash
# SLURM script: noise-strength sweep across many LUMI-G nodes.
#
# This is the "noisy diffqec on LUMI" flagship run.  It uses SLURM's
# job-array feature so each (distance, scale) combination becomes its own
# short job, ideal for filling a queue quickly.
#
# Submit with:
#     sbatch lumi/sweep.slurm           # uses the default account below
#     sbatch --account=project_xxx lumi/sweep.slurm   # override per-job
#
# The number of array tasks is the product of #distances × #scales.
# Default: 2 distances (3, 5) × 6 scales (0.5 .. 2.0) = 12 tasks.

#SBATCH --job-name=noisy-diffqec-sweep
#SBATCH --account=project_465003017
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=lumi/logs/sweep-%A_%a.out
#SBATCH --error=lumi/logs/sweep-%A_%a.err

set -euo pipefail

PROJECT_ROOT="/scratch/project_465003017/diffqec-noisy"
VENV_DIR="${PROJECT_ROOT}/.venv-lumi"
SWEEP_ROOT="${PROJECT_ROOT}/ckpts/sweep-${SLURM_JOB_ID:-manual}"
SCALES_CSV="${SCALES_CSV:-0.5 0.75 1.0 1.25 1.5 2.0}"
DISTANCES_CSV="${DISTANCES_CSV:-3 5}"

mkdir -p "${PROJECT_ROOT}/lumi/logs" "${SWEEP_ROOT}"

# Convert CSVs to bash arrays.
read -r -a SCALES <<< "${SCALES_CSV}"
read -r -a DISTANCES <<< "${DISTANCES_CSV}"
NUM_SCALES=${#SCALES[@]}
NUM_DIST=${#DISTANCES[@]}

DIST_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SCALES ))
SCALE_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SCALES ))
DISTANCE=${DISTANCES[$DIST_IDX]}
SCALE=${SCALES[$SCALE_IDX]}

CKPT_DIR="${SWEEP_ROOT}/d${DISTANCE}_scale${SCALE}"
mkdir -p "${CKPT_DIR}"

echo "[sweep.slurm] task ${SLURM_ARRAY_TASK_ID}  d=${DISTANCE}  scale=${SCALE}  host=$(hostname)"

# Load the currently-supported LUMI stack (LUMI/24.03 is deprecated).
module load LUMI/25.03 2>/dev/null || module load LUMI/24.11 2>/dev/null || true
module load partition/G 2>/dev/null || true
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_ROOT}"

python -m scripts.train_noisy_diffqec \
    --distance   "${DISTANCE}" \
    --rounds     "${ROUNDS:-5}" \
    --shots      "${SHOTS:-15000}" \
    --epochs     "${EPOCHS:-20}" \
    --hidden     "${HIDDEN:-64}" \
    --num-denoiser-layers 4 \
    --noise      "${NOISE:-emerald}" \
    --T          "${T:-100}" \
    --scale      "${SCALE}" \
    --device     cuda \
    --seed       $(( 1000 * DIST_IDX + SCALE_IDX )) \
    --checkpoint-dir "${CKPT_DIR}" \
    --sweep-output "${SWEEP_ROOT}/sweep.json"

echo "[sweep.slurm] task ${SLURM_ARRAY_TASK_ID} done -> ${CKPT_DIR}"
