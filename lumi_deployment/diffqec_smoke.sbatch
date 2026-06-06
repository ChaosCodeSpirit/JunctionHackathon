#!/usr/bin/env bash
# diffqec_smoke.sbatch — DiffQEC smoke test + pytest on LUMI-G.
#
# Runs the Surface Code / DiffQEC pipeline smoke test:
#   1. python -m diffqec.smoke_test  (tiny model train/eval on synthetic d=3 data)
#   2. pytest tests/test_diffqec.py -v  (unit + integration tests)
#
# Outputs go to $REPO_DIR/results/diffqec_smoke/$SLURM_JOB_ID/
#
# Usage on LUMI:
#   sbatch --account="$LUMI_PROJECT_ACCOUNT" diffqec_smoke.sbatch
#   # OR: export SBATCH_ACCOUNT="$LUMI_PROJECT_ACCOUNT" && sbatch diffqec_smoke.sbatch
#   squeue -u $USER
#   cat diffqec_smoke-*.out

#SBATCH --job-name=diffqec_smoke
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --output=diffqec_smoke-%j.out
#SBATCH --error=diffqec_smoke-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default REPO_DIR to repo root (parent of lumi_deployment/)
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/venv}"

echo "== host     : $(hostname)"
echo "== date     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "== account   : ${SLURM_JOB_ACCOUNT:-${SBATCH_ACCOUNT:-<unset — pass --account or export SBATCH_ACCOUNT before sbatch>}}"
echo "== slurm    : job ${SLURM_JOB_ID:-n/a} on ${SLURM_CLUSTER_NAME:-n/a}"
echo "== gpus     : ${SLURM_GPUS:-n/a}"
echo "== repo dir : $REPO_DIR"
echo "== venv dir : $VENV_DIR"

# Module stack — LUMI/25.09
module purge
module load LUMI/25.09
module load partition/G
module load rocm
module load cray-python/3.11.7

echo "== module list:"
module list 2>&1 | sed 's/^/  /'

# Activate venv
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    echo "== venv   : activated $VENV_DIR"
else
    echo "ERROR: venv not found at $VENV_DIR" >&2
    echo "Run bash setup_lumi_env.sh first" >&2
    exit 1
fi

# Output directory
OUT_DIR="$REPO_DIR/results/diffqec_smoke/${SLURM_JOB_ID:-debug}"
mkdir -p "$OUT_DIR"
echo "== output dir: $OUT_DIR"

# Run DiffQEC smoke test
echo ""
echo "== === DiffQEC Smoke Test ==="
set +e
python -m diffqec.smoke_test
SMOKE_EXIT=$?
set -e

echo ""
echo "== smoke test exit code: $SMOKE_EXIT"

# Run pytest
echo ""
echo "== === PyTest: test_diffqec.py ==="
cd "$REPO_DIR"
set +e
python -m pytest tests/test_diffqec.py -v --tb=short 2>&1 | tee "$OUT_DIR/pytest_output.txt"
PYTEST_EXIT=${PIPESTATUS[0]}
set -e

echo ""
echo "== pytest exit code: $PYTEST_EXIT"
echo "== full pytest output: $OUT_DIR/pytest_output.txt"
echo "== results dir: $OUT_DIR"
echo "== listing:"
ls -lh "$OUT_DIR"

# Exit with smoke test status (smoke test is the primary gate)
if [[ $SMOKE_EXIT -ne 0 ]]; then
    echo "== FAIL: diffqec.smoke_test failed (exit $SMOKE_EXIT)"
    exit $SMOKE_EXIT
elif [[ $PYTEST_EXIT -ne 0 ]]; then
    echo "== WARN: pytest had failures (exit $PYTEST_EXIT) — review output"
    exit $PYTEST_EXIT
else
    echo "== ALL CHECKS PASSED"
    exit 0
fi
