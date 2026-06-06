#!/usr/bin/env bash
# hello_smoke.sbatch — 2-minute LUMI-G smoke test.
#
# Run this FIRST after `ssh lumi` to confirm:
#   1. Your account is valid (pass --account="$LUMI_PROJECT_ACCOUNT" or export SBATCH_ACCOUNT)
#   2. ROCm + torch can see the MI250X GCD
#   3. SLURM end-to-end works (submit, allocate, execute, return)
#
# Usage on LUMI:
#   sbatch --account="$LUMI_PROJECT_ACCOUNT" hello_smoke.sbatch
#   # OR: export SBATCH_ACCOUNT="$LUMI_PROJECT_ACCOUNT" && sbatch hello_smoke.sbatch
#   squeue -u $USER
#   cat hello_smoke-*.out
#
# Expected in stdout:
#   - "SMOKE OK"
#   - torch device: cuda (if ROCm working) or cpu
#   - rocminfo gfx lines (if rocminfo available)

#SBATCH --job-name=hello_smoke
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:02:00
#SBATCH --mem=8G
#SBATCH --output=hello_smoke-%j.out
#SBATCH --error=hello_smoke-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/venv}"

echo "== host   : $(hostname)"
echo "== date   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "== slurm  : job ${SLURM_JOB_ID:-n/a} on ${SLURM_CLUSTER_NAME:-n/a}"
echo "== gpus   : ${SLURM_GPUS:-n/a}"
echo "== repo   : $REPO_DIR"

# Module stack — LUMI/25.09 (LUMI/23.09 is deprecated on this system)
module purge
module load LUMI/25.09
module load partition/G
module load rocm
module load cray-python/3.11.7

echo "== module list:"
module list 2>&1 | sed 's/^/  /'

# Activate venv at $REPO_DIR/venv (default)
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    echo "== venv   : activated $VENV_DIR"
else
    echo "== venv   : (no venv found at $VENV_DIR — using system cray-python)"
fi

python3 - <<'PY'
import sys, platform, subprocess
print("== python :", sys.version.split()[0], "on", platform.node())

# Check torch + GPU
try:
    import torch
    print("== torch  :", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("== device :", device)
    if torch.cuda.is_available():
        print("== gpu name:", torch.cuda.get_device_name(0))
        # Quick matmul to verify GPU works
        x = torch.rand(1024, 1024, device=device)
        y = torch.rand(1024, 1024, device=device)
        z = (x @ y).cpu()
        print("== gpu mm : OK, shape", z.shape)
    else:
        print("== gpu    : NOT AVAILABLE (CPU only)")
except Exception as e:
    print("== torch  : NOT AVAILABLE ->", e)

# Check ROCm visibility via rocminfo
try:
    r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=15)
    gpus = [ln.strip() for ln in r.stdout.splitlines() if "gfx" in ln.lower()]
    print("== rocminfo gfx:", gpus[:4] or ["<none>"])
except Exception as e:
    print("== rocminfo: NOT AVAILABLE ->", e)

# Check key QEC imports
for mod in ["stim", "qiskit", "pymatching"]:
    try:
        m = __import__(mod)
        print(f"== {mod}  : {getattr(m, '__version__', 'installed')}")
    except Exception as e:
        print(f"== {mod}  : NOT AVAILABLE -> {e}")

print("== SMOKE OK")
PY