# LUMI Deployment — Day-of Workflow

> Quick card for running Surface Code + DiffQEC jobs on LUMI.
> Assumes you've completed [preflight.md](./preflight.md).

---

## Before You Start

1. Confirm your SSH alias (ask admin — don't assume `lumi`)
2. Confirm your project ID via `lumi-allocations` on LUMI
3. Verify budget: `lumi-allocations | grep -i "gpu-hours"` (or similar)

---

## 1. Sync Repo to LUMI (Laptop)

```bash
# From laptop — sync to LUMI
# Confirm the remote path with your admin. Example using $PROJECT:
cd JunctionHackathon
rsync -avP --exclude='venv/' --exclude='.git/' --exclude='results/' \
  ./ lumi:'$PROJECT/JunctionHackathon/'
```

Or if the repo already exists on LUMI:

```bash
rsync -avP --exclude='venv/' --exclude='.git/' --exclude='results/' \
  ./ lumi:'$PROJECT/JunctionHackathon/'
```

---

## 2. Set Up Environment (On LUMI)

```bash
ssh lumi
cd "$PROJECT/JunctionHackathon/lumi_deployment"

# If first time or venv missing:
bash setup_lumi_env.sh
source ~/.bashrc   # pick up cache redirects to /scratch

# If venv already exists, just activate:
source ../venv/bin/activate
```

---

## 3. Verify Environment

```bash
# Check modules
module list

# Verify imports
python - <<'PY'
import stim, qiskit, pymatching, numpy, torch
print("All key imports OK")
PY

# Check DiffQEC smoke test can run
python -m diffqec.smoke_test --help 2>/dev/null || echo "smoke_test has no --help (normal)"
```

---

## 4. Run Jobs

### Smoke Test (2 min, dev-g partition)

```bash
export LUMI_PROJECT_ACCOUNT=project_<your-id>   # or source lumi_deployment/.env
export SBATCH_ACCOUNT="$LUMI_PROJECT_ACCOUNT"
sbatch --account="$LUMI_PROJECT_ACCOUNT" hello_smoke.sbatch
squeue -u $USER
cat hello_smoke-*.out
```

Expected: `SMOKE OK`

### DiffQEC Smoke Test (10 min, dev-g partition)

```bash
sbatch --account="$LUMI_PROJECT_ACCOUNT" diffqec_smoke.sbatch
squeue -u $USER
cat diffqec_smoke-*.out
```

Expected:
- `SMOKE TEST PASSED`
- `tests/test_diffqec.py::... PASSED`

---

## 5. Collect Results

Results land in `$PROJECT_DIR/results/<job_name>/<job_id>/`.

```bash
# On LUMI — list results
ls -lh "$PROJECT/JunctionHackathon/results/"

# On laptop — rsync results back (confirm remote path with admin)
rsync -avP lumi:'$PROJECT/JunctionHackathon/results/' ./results/
```

---

## 6. Monitor and Debug

```bash
# List running jobs
squeue -u $USER

# Check recent job output
cat diffqec_smoke-*.out

# Check for errors
cat diffqec_smoke-*.err

# If job failed, common causes:
# - out of budget      → lumi-allocations
# - module load fail   → module avail; check LUMI/25.09
# - import error       → re-run setup_lumi_env.sh; only remove venv after confirming its path
# - partition wrong    → use dev-g for short jobs, standard-g for longer GPU work
# - home quota hit    → bash lumi_deployment/cleanup_lumi.sh --dry-run (see storage.md)
```

---

## 7. Cancel a Job

```bash
scancel <jobid>
# Or cancel all your jobs:
scancel -u $USER
```

---

## 8. Daily Workflow Summary

```
laptop: rsync to lumi
lumi:   cd "$PROJECT/JunctionHackathon/lumi_deployment"
lumi:   source ../venv/bin/activate
lumi:   sbatch --account="$LUMI_PROJECT_ACCOUNT" hello_smoke.sbatch   # quick check
lumi:   sbatch --account="$LUMI_PROJECT_ACCOUNT" diffqec_smoke.sbatch  # main smoke test
laptop: rsync results back
```

---

## Key Paths

| Path | Description |
|---|---|
| `$REPO_DIR` / `$PROJECT_DIR` | Repo root on LUMI (normally `$PROJECT/JunctionHackathon`) |
| `$PROJECT_DIR/venv` | Symlink → `/scratch/$USER/venv-junction` (actual venv on scratch) |
| `$PROJECT_DIR/results/` | Job outputs (rsync to laptop after runs) |
| `$PROJECT_DIR/lumi_deployment/` | Deployment scripts |
| `/scratch/$USER/` | Per-user scratch: venvs, caches, active data (no SBU, 90-day purge) |
| `/scratch/$USER/cache/` | Pip/uv/hf caches (re-pointed by `setup_lumi_env.sh`) |

> **Storage quota hit?** See [storage.md](./storage.md) for cleanup and prevention.

---

## Safety Rules

1. **Never run GPU work on login nodes** — always use `sbatch`
2. **Never hardcode project IDs** — use `${SLURM_JOB_ACCOUNT}`
3. **Never put large data in `$HOME`** — use `$PROJECT` or `$SCRATCH`
4. **Never build a venv in `$HOME`** — `setup_lumi_env.sh` puts it on `/scratch/$USER`; if you create one manually, build it there too
5. **Confirm SSH aliases** before every `ssh`/`rsync`

---

## Quick Reference Card

```
# Sync
rsync -avP ./ lumi:'$PROJECT/JunctionHackathon/'

# Submit
export SBATCH_ACCOUNT="$LUMI_PROJECT_ACCOUNT"
sbatch --account="$LUMI_PROJECT_ACCOUNT" hello_smoke.sbatch
sbatch --account="$LUMI_PROJECT_ACCOUNT" diffqec_smoke.sbatch

# Monitor
squeue -u $USER
cat hello_smoke-*.out
cat diffqec_smoke-*.out

# Collect
rsync -avP lumi:'$PROJECT/JunctionHackathon/results/' ./results/
```

For more details, see [README.md](./README.md), [preflight.md](./preflight.md), and [storage.md](./storage.md).