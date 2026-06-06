#!/usr/bin/env bash
# submit_54q.sh — local (Windows / Linux) helper that rsyncs the LUMI
# benchmark files to LUMI and submits the 54-qubit DiffQEC job.
#
# Prerequisites:
#   1. `ssh lumi` works (alias added to ~/.ssh/config on this machine;
#      see Memory/infrastructure/lumi/lumi-quantumhack-2026.md)
#   2. PROJECT_ID is set in .env (see .env.example) OR exported in your shell
#
# Usage:
#   cd lumi
#   cp .env.example .env       # then edit .env with PROJECT_ID
#   ./submit_54q.sh push       # rsync code to LUMI only
#   ./submit_54q.sh submit     # push + sbatch the 54-qubit job
#   ./submit_54q.sh status     # squeue on LUMI
#   ./submit_54q.sh fetch      # rsync benchmark_results.json back to laptop
#   ./submit_54q.sh logs       # tail the latest job log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The repo root is the parent of the lumi/ folder
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REMOTE="${REMOTE:-lumi:~/junctionhackathon-lumi}"

# Load .env if present
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; . "$SCRIPT_DIR/.env"; set +a
fi

: "${PROJECT_ID:?Set PROJECT_ID in .env (e.g. project_465003017) or export it}"
: "${REMOTE_DIR:=~/junctionhackathon-lumi}"

sync_up() {
  echo ">> rsync $REPO_DIR -> $REMOTE"
  # Sync the whole repo (small), excluding caches, venv, .git
  rsync -avP --delete \
    --exclude '.git' --exclude '__pycache__' --exclude 'venv' \
    --exclude '.pytest_cache' --exclude '*.pyc' --exclude '.ruff_cache' \
    --exclude '.uv' --exclude 'uv.lock' \
    "$REPO_DIR/" "$REMOTE/"
  # Ensure lumi-specific files are at the top level of REMOTE_DIR
  rsync -avP "$SCRIPT_DIR/diffqec_54q.sh" "$REMOTE/"
  rsync -avP "$SCRIPT_DIR/requirements_lumi.txt" "$REMOTE/"
  echo ">> remote dir: $REMOTE_DIR"
}

run_remote() {
  ssh lumi "PROJECT_DIR=$REMOTE_DIR SLURM_JOB_ACCOUNT=$PROJECT_ID bash -s" -- "$@"
}

submit_job() {
  run_remote "cd $REMOTE_DIR && sbatch diffqec_54q.sh"
}

show_status() {
  run_remote 'squeue -u $USER -o "%.10i %.9P %.8j %.8u %.2t %.10M %.6D %.20R %.8C %.10m" || true'
}

fetch_results() {
  echo ">> rsync $REMOTE:benchmark_results.json -> $SCRIPT_DIR/"
  rsync -avP "$REMOTE/benchmark_results.json" "$SCRIPT_DIR/" || true
  rsync -avP "$REMOTE/diffqec_54q_*.log" "$SCRIPT_DIR/logs/" 2>/dev/null || \
    mkdir -p "$SCRIPT_DIR/logs" && rsync -avP "$REMOTE/diffqec_54q_*.log" "$SCRIPT_DIR/logs/"
}

show_logs() {
  run_remote "cd $REMOTE_DIR && ls -lht diffqec_54q_*.log 2>/dev/null | head -5 && echo '---LATEST---' && tail -200 diffqec_54q_\$(ls -t diffqec_54q_*.log 2>/dev/null | head -1 | cut -d_ -f3 | cut -d. -f1).log 2>/dev/null || true"
}

case "${1:-help}" in
  push)    sync_up ;;
  submit)  sync_up; submit_job ;;
  status)  show_status ;;
  fetch)   fetch_results ;;
  logs)    show_logs ;;
  all)     sync_up; submit_job; show_status ;;
  *)
    cat <<EOF
Usage: $0 <command>
  push     rsync repo to LUMI (~\$REMOTE_DIR)
  submit   push + sbatch diffqec_54q.sh
  status   show squeue on LUMI
  fetch    rsync benchmark_results.json + logs back to laptop
  logs     tail the latest job log on LUMI
  all      push + submit + status

Env (set in lumi/.env):
  PROJECT_ID=project_465003017   # required
  REMOTE_DIR=~/junctionhackathon-lumi
EOF
    exit 1 ;;
esac
