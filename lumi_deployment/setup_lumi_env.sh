#!/usr/bin/env bash
# setup_lumi_env.sh — bootstrap the JunctionHackathon venv on LUMI-G.
# Verified working on LUMI/25.09 stack.
#
# Storage note (LUMI-G):
#   The venv is built at /scratch/$USER/venv-junction by default — never on
#   $HOME — to stay under LUMI's home file-count quota (100k files on
#   LUMI-G).  A symlink is left at $PROJECT_DIR/venv so existing
#   `source ../venv/bin/activate` calls keep working.
#   Pip/uv/jupyter/hf caches are also re-pointed at /scratch/$USER/cache
#   (see the .bashrc write at the end of this script).
#
# Usage:
#   bash setup_lumi_env.sh
#   VENV_DIR=/some/other/path bash setup_lumi_env.sh     # override location
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default PROJECT_DIR to repo root (parent of lumi_deployment/)
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
USER_NAME="${USER:-$(id -un)}"
# Default venv lives on scratch, NOT on home.  Override with VENV_DIR=... if
# you really need to put it somewhere else.
VENV_DIR="${VENV_DIR:-/scratch/${USER_NAME}/venv-junction}"
# Find requirements.txt at repo root
REQ_FILE="${REQ_FILE:-$PROJECT_DIR/requirements.txt}"

echo "== setup_lumi_env.sh"
echo "== project dir : $PROJECT_DIR"
echo "== venv dir    : $VENV_DIR"
echo "== req file    : $REQ_FILE"

# LUMI/25.09 is the current supported software stack.
# (LUMI/23.09 is deprecated and rejects with cray-python/3.11.5 missing.)
module --force purge
module load LUMI/25.09
module load partition/G
module load rocm
module load cray-python/3.11.7

echo "== module stack:"
module list 2>&1 | sed 's/^/  /'

# Build the venv with --system-site-packages so we pick up Cray's
# optimized numpy / mpi4py etc.
if [[ -d "$VENV_DIR" ]]; then
    echo "== venv already exists at $VENV_DIR — activating and updating"
else
    echo "== creating venv at $VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    python3 -m venv "$VENV_DIR" --system-site-packages
fi

# Symlink $PROJECT_DIR/venv -> $VENV_DIR so the conventional
# `source ../venv/bin/activate` keeps working without copying the
# 30–50k small venv files into $HOME (which would blow the file-count
# quota).  If the user already has a real directory at $PROJECT_DIR/venv
# from a prior install, we move it aside and symlink instead.
PROJ_VENV="$PROJECT_DIR/venv"
if [[ "$(readlink -f "$PROJ_VENV" 2>/dev/null || true)" == "$(readlink -f "$VENV_DIR" 2>/dev/null || true)" ]]; then
    echo "== project-venv symlink already points to $VENV_DIR"
elif [[ -d "$PROJ_VENV" && ! -L "$PROJ_VENV" ]]; then
    echo "== migrating existing $PROJ_VENV (was on \$HOME) to $VENV_DIR"
    # Move the existing venv into the scratch location, then symlink back.
    mv "$PROJ_VENV" "$VENV_DIR"
    ln -s "$VENV_DIR" "$PROJ_VENV"
elif [[ ! -e "$PROJ_VENV" ]]; then
    echo "== symlinking $PROJ_VENV -> $VENV_DIR"
    ln -s "$VENV_DIR" "$PROJ_VENV"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "== upgrading pip"
python -m pip install --upgrade pip setuptools wheel

echo "== installing requirements from $REQ_FILE"
python -m pip install -r "$REQ_FILE"

# Verify key imports for Surface Code / DiffQEC pipeline
echo "== verifying key imports..."
python - <<'PY'
import importlib
# Core QEC stack
mods = ["qiskit", "stim", "pymatching", "numpy"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  OK: {m} ({ver})")
    except ImportError as e:
        print(f"  FAIL: {m} -> {e}")

# DiffQEC / ML stack (torch may be installed via requirements)
try:
    import torch
    print(f"  OK: torch ({torch.__version__}) device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
except ImportError as e:
    print(f"  FAIL: torch -> {e}")
PY

echo "== environment ready at $VENV_DIR"
echo "== next step: sbatch hello_smoke.sbatch"

# Point transient caches at /scratch so they don't push $HOME over the
# file-count quota.  Idempotent — re-running won't double up the block.
CACHE_ROOT="/scratch/${USER_NAME}/cache"
mkdir -p "$CACHE_ROOT"/{pip,uv,huggingface,matplotlib,jupyter,triton} 2>/dev/null || true

CACHE_BEGIN="# >>> junction-hackathon cache env (managed by setup_lumi_env.sh) >>>"
CACHE_END="# <<< junction-hackathon cache env <<<"
BASHRC="$HOME/.bashrc"
if [[ -f "$BASHRC" ]] && grep -qF "$CACHE_BEGIN" "$BASHRC"; then
    echo "== cache-redirect block already in $BASHRC"
else
    echo "== appending cache-redirect block to $BASHRC"
    cat >> "$BASHRC" <<EOF

$CACHE_BEGIN
export XDG_CACHE_HOME=$CACHE_ROOT
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export UV_CACHE_DIR=$CACHE_ROOT/uv
export HF_HOME=$CACHE_ROOT/huggingface
export MPLCONFIGDIR=$CACHE_ROOT/matplotlib
export JUPYTER_RUNTIME_DIR=$CACHE_ROOT/jupyter
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
$CACHE_END
EOF
fi