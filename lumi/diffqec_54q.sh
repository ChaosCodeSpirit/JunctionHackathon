#!/bin/bash
#SBATCH --job-name=diffqec_54q
#SBATCH --account=project_465003017
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=diffqec_54q_%j.log
#SBATCH --error=diffqec_54q_%j.err

# 54-qubit DiffQEC benchmark on LUMI (Junction QuantumHack 2026)
#
# This script:
#   1. Sets up the LUMI/25.09 + cray-python module stack.
#   2. Creates a venv with qiskit, qiskit-aer, stim, pymatching, torch.
#   3. Generates a 54-qubit surface-code syndrome dataset (Qiskit Aer stabilizer).
#   4. Trains a small DiffQEC model on the syndromes.
#   5. Evaluates DiffQEC vs the trivial "no-correction" baseline at
#      multiple physical error rates p.
#
# The 54 qubits mirror IQM Q50 / VTT QX (Q50 has 54 physical qubits numbered
# QB1..QB54).  The benchmark uses a rotated surface code d=5 (49 qubits)
# plus 5 flag ancilla = 54 total.

set -euo pipefail

echo "== host   : $(hostname)"
echo "== date   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "== account: ${SLURM_JOB_ACCOUNT:-<unset>}"
echo "== slurm  : job ${SLURM_JOB_ID} on ${SLURM_CLUSTER_NAME}"

# ---- Module stack (LUMI/25.09 — verified 2026-06-05) ----
module --force purge
module load LUMI/25.09
module load partition/C
module load cray-python/3.11.7

module list 2>&1 | sed 's/^/  /'

# ---- Working directory on LUMI-P (Lustre scratch) ----
# /users/$USER has a 1 GB quota — too small for torch (~800 MB).
# Use the project's scratch space on LUMI-P instead.  The project source
# is rsynced into ~/junctionhackathon-lumi/ by the laptop-side wrapper;
# we just point PYTHONPATH at it.
PROJ_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT:-project_465003017}"
WORKDIR="${PROJ_SCRATCH}/${USER}/junctionhackathon-lumi"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
echo "== workdir: ${WORKDIR}"

# ---- Python environment (cray-python/3.11.7) ----
# We install into a project-local site-packages dir on LUMI-P (Lustre).
# cray-python ships pip 23.2.1 system-wide; we just use it directly.
# The project source is rsynced into $HOME/junctionhackathon-lumi/ by the
# laptop-side wrapper; we just point PYTHONPATH at it.
SITE="${WORKDIR}/site-packages"
mkdir -p "${SITE}"
PROJ_SRC="${HOME}/junctionhackathon-lumi"
export PYTHONPATH="${PROJ_SRC}:${SITE}:${PYTHONPATH:-}"

# If torch isn't already installed in the site dir, do the install.
if ! python -c "import torch" 2>/dev/null; then
  echo ">> Installing project dependencies into ${SITE} (this takes ~3-5 min once)..."
  # Use --no-cache-dir to keep pip's /tmp usage small
  # Build into a /tmp staging dir first, then move to Lustre
  STAGE="/tmp/pip-stage-$$"
  mkdir -p "${STAGE}"
  python -m pip install --quiet --no-cache-dir --target="${STAGE}" \
      "qiskit==2.4.1" \
      "qiskit-aer==0.17.2" \
      "stim==1.16.0" \
      "pymatching==2.4.0" \
      "numpy==2.4.4" \
      "torch>=2.0" 2>&1 | tail -20
  # Now rsync the staging dir to the Lustre site-packages
  echo ">> Moving installed packages from /tmp to Lustre site-packages..."
  rsync -a "${STAGE}/" "${SITE}/" 2>&1 | tail -5
  rm -rf "${STAGE}"
else
  echo ">> Reusing existing site-packages at ${SITE}"
fi

# Verify torch is importable
python -c "import torch, qiskit, qiskit_aer, stim, pymatching, numpy; print('Imports OK:', torch.__version__, qiskit.__version__, qiskit_aer.__version__)"

# ---- Run the benchmark ----
echo "=="
echo "== Running 54-qubit DiffQEC benchmark"
echo "=="

# 1. Train a small DiffQEC model (if no checkpoint)
CKPT="${WORKDIR}/diffqec_54q.pt"
if [[ ! -f "${CKPT}" ]]; then
  echo ">> Training DiffQEC on d=5 surface-code syndromes (p=1e-3)..."
  python -m benchmarks.q50_54q.train_diffqec \
      --p 0.001 \
      --shots 8192 \
      --epochs 30 \
      --hidden 64 \
      --n_layers 3 \
      --out "${CKPT}" 2>&1 | tail -20
else
  echo ">> Reusing existing checkpoint ${CKPT}"
fi

# 2. Run the benchmark across multiple error rates
echo ">> Running benchmark across p ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}..."
python -m benchmarks.q50_54q.benchmark \
    --p_list 0.0001 0.0003 0.001 0.003 0.01 \
    --shots 8192 \
    --model_path "${CKPT}" \
    --out "benchmark_results.json" 2>&1

echo "=="
echo "== DONE"
echo "=="
ls -l benchmark_results.json 2>/dev/null || true
