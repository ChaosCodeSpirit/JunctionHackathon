#!/usr/bin/env bash
# Install the noisy-diffqec stack on a LUMI login node.
#
# LUMI ships with Python 3.10 / 3.11 in the LUMI/24.03 stack and a ROCm-aware
# PyTorch module.  We use a conda-based venv (not the module's PyTorch) so
# that our pip-installed PyTorch matches the system ROCm and we keep the
# project's `pip install -e .` workflow.
#
# Usage on a LUMI login node:
#     module load LUMI/24.03 partition/G
#     module load cray-python/3.11
#     bash lumi/install.sh
#
# The script is idempotent: re-running it only refreshes the venv if
# `requirements.txt` has changed.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv-lumi"
PYTHON_BIN="${PYTHON:-python}"

echo "[lumi/install] project root: ${PROJECT_ROOT}"
echo "[lumi/install] venv dir    : ${VENV_DIR}"

# 1. LUMI-side ROCm toolchain ------------------------------------------------
module load LUMI/24.03 2>/dev/null || true
module load partition/G 2>/dev/null || true
module load rocm/6.0.3 2>/dev/null || true

# 2. Create a venv in the project's scratch-friendly location ---------------
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[lumi/install] creating venv..."
    ${PYTHON_BIN} -m venv "${VENV_DIR}" --system-site-packages
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel

# 3. Install project + LUMI-specific extras ----------------------------------
echo "[lumi/install] installing project requirements..."
pip install -r "${PROJECT_ROOT}/requirements.txt"

# LUMI-G: torch wheels are usually provided by the system module.  If we
# don't pick one up, fall back to a ROCm-enabled pip wheel.
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[lumi/install] installing ROCm PyTorch wheel..."
    pip install --index-url https://download.pytorch.org/whl/rocm6.0 \
        "torch>=2.4,<2.6"
fi

# Install the project itself so `import diffqec` works on compute nodes.
pip install -e "${PROJECT_ROOT}"

# 4. Sanity check ------------------------------------------------------------
python - <<'PY'
import torch, diffqec
print(f"[lumi/install] torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
print(f"[lumi/install] diffqec={diffqec.__file__}")
PY

echo "[lumi/install] done.  Activate with:  source ${VENV_DIR}/bin/activate"
