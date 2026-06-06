#!/usr/bin/env bash
# cleanup_lumi.sh — reclaim LUMI home-directory file-count and space quota.
#
# Why this exists:
#   The JunctionHackathon pipeline (Qiskit + Stim + PyMatching + DiffQEC) tends to
#   leave behind tens of thousands of small files in $HOME: __pycache__, pip/uv
#   wheels, ipython history, jupyter runtimes, .pytest_cache, etc.  On LUMI-G
#   the home directory is hard-capped at ~50 GB / 100,000 files, and Lustre is
#   especially slow with many small files — so the second a user exceeds the
#   count quota, every subsequent login warning becomes an obstacle.
#
# What it does (default, safe):
#   - Removes Python debris: __pycache__/, .pytest_cache/, *.pyc, .coverage
#   - Removes pip / uv caches under ~/.cache
#   - Removes matplotlib / jupyter / ipython runtime junk
#   - Re-points pip/uv at /scratch/$USER/cache (idempotent block in ~/.bashrc)
#
# What it does NOT touch:
#   - Any venv (./venv, ./.venv, /scratch/$USER/venv-junction)
#   - .git, source code, results/, lumi_deployment/, requirements files
#   - Anything in /scratch, /project, /appl (out of home via -xdev)
#
# Usage:
#   bash lumi_deployment/cleanup_lumi.sh --dry-run   # show what would happen
#   bash lumi_deployment/cleanup_lumi.sh             # safe default cleanup
#   bash lumi_deployment/cleanup_lumi.sh --deep      # also rm -rf ~/.cache
#                                                    # use only if still over
#                                                    # quota after default run
#
# Always run with --dry-run first.  Nothing here is recoverable.

set -euo pipefail

DRY_RUN=0
DEEP=0
REPOINT_CACHES=1

print_usage() {
  cat <<'EOF'
Usage: bash lumi_deployment/cleanup_lumi.sh [--dry-run] [--deep] [--no-repoint]

  --dry-run      Print every action as [dry-run]; change nothing.
  --deep         Also nuke ~/.cache entirely.  Use only if default is
                 not enough — caches regenerate on next use.
  --no-repoint   Skip writing the cache-redirect block to ~/.bashrc.

Default: removes Python debris and pip/uv/ipython caches, then writes a
small idempotent block to ~/.bashrc that re-points future caches to
/scratch/$USER/cache.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --dry-run|-n)   DRY_RUN=1 ;;
    --deep)         DEEP=1 ;;
    --no-repoint)   REPOINT_CACHES=0 ;;
    -h|--help)      print_usage; exit 0 ;;
    *)              echo "Unknown argument: $arg" >&2; print_usage >&2; exit 2 ;;
  esac
done

HOME_DIR="${HOME:-/tmp}"
USER_NAME="${USER:-$(id -un)}"
SCRATCH_USER="/scratch/${USER_NAME}/cache"

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  [dry-run] %s\n' "$*"
  else
    printf '  [run]     %s\n' "$*"
    eval "$@"
  fi
}

files_in_home() {
  find "$HOME_DIR" -xdev -type f 2>/dev/null | wc -l
}

du_h() {
  du -sh "$1" 2>/dev/null | cut -f1 || echo "?"
}

echo "== LUMI home cleanup =="
echo "  user      : $USER_NAME"
echo "  home      : $HOME_DIR"
echo "  scratch   : $SCRATCH_USER"
echo "  mode      : $([ "$DRY_RUN" -eq 1 ] && echo 'DRY-RUN' || echo 'LIVE')"
echo "  deep      : $DEEP"
echo "  repoint   : $REPOINT_CACHES"
echo

echo "== Snapshot BEFORE =="
files_in_home | awk '{printf "  home file count: %s\n", $1}'
[ -d "$HOME_DIR/.cache" ] && printf "  ~/.cache      : %s\n" "$(du_h "$HOME_DIR/.cache")" || true
[ -d "$HOME_DIR/.local" ] && printf "  ~/.local      : %s\n" "$(du_h "$HOME_DIR/.local")" || true
[ -d "$HOME_DIR/.ipython" ] && printf "  ~/.ipython    : %s\n" "$(du_h "$HOME_DIR/.ipython")" || true
if command -v lfs >/dev/null 2>&1; then
  lfs quota -u "$USER_NAME" "$HOME_DIR" 2>/dev/null | sed 's/^/  lfs: /' || true
fi
echo

# --- 1. Python debris --------------------------------------------------------
echo "== Removing Python debris =="
find "$HOME_DIR" -xdev -type d -name '__pycache__' -prune -print 2>/dev/null \
  | while IFS= read -r d; do [ -n "$d" ] && run rm -rf "$d"; done
find "$HOME_DIR" -xdev -type d -name '.pytest_cache' -prune -print 2>/dev/null \
  | while IFS= read -r d; do [ -n "$d" ] && run rm -rf "$d"; done
find "$HOME_DIR" -xdev -type d -name '.mypy_cache' -prune -print 2>/dev/null \
  | while IFS= read -r d; do [ -n "$d" ] && run rm -rf "$d"; done
find "$HOME_DIR" -xdev -type d -name '.ruff_cache' -prune -print 2>/dev/null \
  | while IFS= read -r d; do [ -n "$d" ] && run rm -rf "$d"; done
find "$HOME_DIR" -xdev -type d -name '.ipynb_checkpoints' -prune -print 2>/dev/null \
  | while IFS= read -r d; do [ -n "$d" ] && run rm -rf "$d"; done
find "$HOME_DIR" -xdev -type f -name '*.pyc' -print -delete 2>/dev/null | sed 's/^/  [run]     rm -f /'
find "$HOME_DIR" -xdev -type f -name '.coverage' -print -delete 2>/dev/null | sed 's/^/  [run]     rm -f /'
echo

# --- 2. Tool caches ----------------------------------------------------------
echo "== Removing tool caches =="
for d in \
  "$HOME_DIR/.cache/pip" \
  "$HOME_DIR/.cache/uv" \
  "$HOME_DIR/.cache/huggingface" \
  "$HOME_DIR/.cache/matplotlib" \
  "$HOME_DIR/.cache/jupyter" \
  "$HOME_DIR/.cache/torch" \
  "$HOME_DIR/.cache/triton" \
  "$HOME_DIR/.cache/pytest" \
  "$HOME_DIR/.cache/Cython" \
  ; do
  [ -d "$d" ] && run rm -rf "$d"
done
[ -d "$HOME_DIR/.ipython" ] && run rm -rf "$HOME_DIR/.ipython"
[ -d "$HOME_DIR/.local/share/jupyter/runtime" ] \
  && run rm -rf "$HOME_DIR/.local/share/jupyter/runtime"
[ -d "$HOME_DIR/.jupyter/runtime" ] \
  && run rm -rf "$HOME_DIR/.jupyter/runtime"
[ -d "$HOME_DIR/.config/pip" ] && run rm -rf "$HOME_DIR/.config/pip"
[ -d "$HOME_DIR/.config/uv" ]  && run rm -rf "$HOME_DIR/.config/uv"
echo

# --- 3. Optional deep wipe ---------------------------------------------------
if [ "$DEEP" -eq 1 ]; then
  echo "== DEEP wipe of ~/.cache =="
  [ -d "$HOME_DIR/.cache" ] && run rm -rf "$HOME_DIR/.cache"
  echo
fi

# --- 4. Re-point future caches to scratch -----------------------------------
if [ "$REPOINT_CACHES" -eq 1 ]; then
  echo "== Writing cache-redirect block to ~/.bashrc =="
  mkdir -p "$SCRATCH_USER"/{pip,uv,huggingface,matplotlib,jupyter,triton} \
    2>/dev/null || true
  BLOCK_BEGIN="# >>> junction-hackathon cache env (managed by cleanup_lumi.sh) >>>"
  BLOCK_END="# <<< junction-hackathon cache env <<<"
  BASHRC="$HOME_DIR/.bashrc"

  if [ -f "$BASHRC" ] && grep -qF "$BLOCK_BEGIN" "$BASHRC"; then
    echo "  block already present in $BASHRC — leaving as-is"
  else
    if [ "$DRY_RUN" -eq 1 ]; then
      printf '  [dry-run] append cache-redirect block to %s\n' "$BASHRC"
    else
      cat >> "$BASHRC" <<EOF

$BLOCK_BEGIN
# Point transient caches at /scratch (no SBU, not subject to home file-count quota).
export XDG_CACHE_HOME=$SCRATCH_USER
export PIP_CACHE_DIR=$SCRATCH_USER/pip
export UV_CACHE_DIR=$SCRATCH_USER/uv
export HF_HOME=$SCRATCH_USER/huggingface
export MPLCONFIGDIR=$SCRATCH_USER/matplotlib
export JUPYTER_RUNTIME_DIR=$SCRATCH_USER/jupyter
export TRITON_CACHE_DIR=$SCRATCH_USER/triton
$BLOCK_END
EOF
      echo "  appended cache-redirect block to $BASHRC"
      echo "  it takes effect in new shells; run 'source ~/.bashrc' to apply now"
    fi
  fi
  echo
fi

# --- 5. Snapshot AFTER -------------------------------------------------------
echo "== Snapshot AFTER =="
files_in_home | awk '{printf "  home file count: %s\n", $1}'
[ -d "$HOME_DIR/.cache" ] && printf "  ~/.cache      : %s\n" "$(du_h "$HOME_DIR/.cache")" || echo "  ~/.cache      : (gone)"
[ -d "$HOME_DIR/.local" ] && printf "  ~/.local      : %s\n" "$(du_h "$HOME_DIR/.local")" || true
[ -d "$HOME_DIR/.ipython" ] && printf "  ~/.ipython    : %s\n" "$(du_h "$HOME_DIR/.ipython")" || echo "  ~/.ipython    : (gone)"
if command -v lfs >/dev/null 2>&1; then
  lfs quota -u "$USER_NAME" "$HOME_DIR" 2>/dev/null | sed 's/^/  lfs: /' || true
fi
echo
echo "== done =="
echo "  next: open a new shell (or 'source ~/.bashrc') to pick up cache redirects."
echo "  if file-count is still over the limit, re-run with --deep."