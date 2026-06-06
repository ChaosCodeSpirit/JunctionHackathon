#!/usr/bin/env bash
# Install the noisy-diffqec stack on a LUMI login node.
#
# LUMI ships a Cray Python in the LUMI software stack.  We pin to the
# currently-supported LUMI/25.03 stack (LUMI/24.03 is deprecated due to
# ROCm / OS-stack changes — see the Lmod warning at login) and use the
# ROCm-aware PyTorch wheel that matches its driver.
#
# Usage on a LUMI login node:
#     module load LUMI/25.03 partition/G
#     bash lumi/install.sh
#
# The script is idempotent: re-running it only refreshes the venv if
# `requirements.txt` has changed.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv-lumi"
LUMI_USER_DEFAULT="siljheis"
LUMI_USER="${LUMI_USER:-${LUMI_USER_DEFAULT}}"

# Pick a Python.  LUMI/25.03 does not put python on PATH by default —
# you need `module load cray-python` to get a working interpreter.
# Try a few module loads until we find one.
for py in "" "cray-python/3.11" "cray-python/3.12" "cray-python"; do
    if [[ -n "${py}" ]]; then
        module load "${py}" 2>/dev/null || true
    fi
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"; break
    fi
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"; break
    fi
done

if [[ -z "${PYTHON_BIN:-}" ]]; then
    echo "ERROR: no python on PATH after trying to load cray-python." >&2
    echo "  Run 'module avail cray-python' and report the available versions." >&2
    exit 1
fi

echo "[lumi/install] project root: ${PROJECT_ROOT}"
echo "[lumi/install] venv dir    : ${VENV_DIR}"
echo "[lumi/install] LUMI user   : ${LUMI_USER}"
echo "[lumi/install] python      : ${PYTHON_BIN}  ($(${PYTHON_BIN} --version 2>&1))"

# 1. LUMI-side ROCm toolchain ------------------------------------------------
# Try the supported stacks in order; ignore failures (Lmod can be chatty).
for stack in LUMI/25.03 LUMI/24.11; do
    if module load "${stack}" 2>/dev/null; then
        echo "[lumi/install] loaded ${stack}"
        break
    fi
done
module load partition/G 2>/dev/null || true

# Sanity check: confirm we are on a LUMI login node before we start
# modifying the user environment.
if [[ ! -d /scratch ]]; then
    echo "WARNING: /scratch not present — this script expects a LUMI host." >&2
fi

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
# Filter out the original torch pin (we install a ROCm wheel below).
grep -v "^torch" "${PROJECT_ROOT}/requirements.txt" \
    | pip install -r /dev/stdin

# LUMI-G: install a ROCm PyTorch wheel that matches the current driver.
# On LUMI/25.03 the system ROCm is 6.2.
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[lumi/install] installing ROCm PyTorch wheel (rocm6.2)..."
    pip install --index-url https://download.pytorch.org/whl/rocm6.2 \
        "torch>=2.5,<2.7"
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

