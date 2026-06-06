# LUMI Preflight Checklist

> First-time setup walkthrough for JunctionHackathon / Surface Code + DiffQEC on LUMI.
> Estimated time: **~30 minutes** for a new user.

---

## Step 0: Prerequisites

- [ ] You have a CSC account (or can request one via your institution)
- [ ] You have been added to the JunctionHackathon LUMI allocation
- [ ] You have a laptop with an SSH client and `rsync` available
- [ ] You know your SSH alias for LUMI (ask your admin — do NOT assume `lumi`)

---

## Step 1: SSH Access

```bash
# Test login (replace 'lumi' with your confirmed alias)
ssh lumi
# Accept host key if prompted
exit
```

Verify you can log in without password failures. If you get a permission denied error, contact your CSC project administrator.

---

## Step 2: Confirm Allocation

On LUMI (after ssh):

```bash
lumi-allocations
```

This prints your project ID (looks like `project_465001234`). Copy it — you'll need it for Step 4.

---

## Step 3: Configure SSH Config (Laptop)

Add to `~/.ssh/config`:

```
Host lumi
    HostName lumi.csc.fi
    User <your-csc-username>
    IdentityFile ~/.ssh/id_ed25519   # your SSH key
    StrictHostKeyChecking accept-new

Host lumidt
    HostName lumidt.csc.fi
    User <your-csc-username>
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking accept-new
```

> **Note:** SSH aliases `lumi` and `lumidt` are common but confirm with your admin. Do not assume these names.

---

## Step 4: Set Up Local Environment Template

On your **laptop** (not LUMI):

```bash
cd JunctionHackathon/lumi_deployment
cp env.example .env
# Edit .env with your project ID from Step 2
```

In `.env`:
```
export LUMI_PROJECT_ACCOUNT=project_<your-id>
export SBATCH_ACCOUNT="$LUMI_PROJECT_ACCOUNT"
```

> **Warning:** `.env` is gitignored. Never commit it.

---

## Step 5: Bootstrap Environment on LUMI

```bash
# From your laptop, sync the repo into the project filesystem.
# Confirm the remote path first; this example lets the remote shell expand $PROJECT.
rsync -avP --exclude='venv/' --exclude='.git/' --exclude='results/' \
  ./ lumi:'$PROJECT/JunctionHackathon/'

# Then SSH to LUMI and bootstrap from the synced repo.
ssh lumi
cd "$PROJECT/JunctionHackathon/lumi_deployment"
bash setup_lumi_env.sh
```

`setup_lumi_env.sh` will:
1. Purge existing modules
2. Load LUMI/25.09 + partition/G + rocm + cray-python/3.11.7
3. Create or update a venv at `/scratch/$USER/venv-junction` with `--system-site-packages` and symlink it to `$PROJECT_DIR/venv`
4. Install requirements from `requirements.txt`
5. Verify key imports: `qiskit`, `stim`, `pymatching`, `numpy`, `torch`
6. Write an idempotent cache-redirect block to `~/.bashrc` so pip/uv/hf caches go to `/scratch/$USER/cache` instead of `$HOME`

> **Why scratch?** A Python venv contains 30–50 k small files. LUMI-G caps your home at 100 k files total. Putting the venv on scratch avoids blowing the quota. The symlink means `source ../venv/bin/activate` still works. See [storage.md](./storage.md) for details.

After bootstrapping, activate the cache redirects:
```bash
source ~/.bashrc   # pick up the cache-redirect exports
```

Expected output:
```
Environment ready at /scratch/<user>/venv-junction
```

---

## Step 6: Run Smoke Test

```bash
# Set account from your project id, or source lumi_deployment/.env
export LUMI_PROJECT_ACCOUNT=project_<your-id>
export SBATCH_ACCOUNT="$LUMI_PROJECT_ACCOUNT"

# Submit minimal smoke test
sbatch --account="$LUMI_PROJECT_ACCOUNT" hello_smoke.sbatch

# Watch it
squeue -u $USER

# Check output when done
cat hello_smoke-*.out
```

Expected in output:
- `SMOKE OK`
- torch device: cuda (if ROCm working) or cpu

---

## Step 7: Run DiffQEC Smoke Test

```bash
sbatch --account="$LUMI_PROJECT_ACCOUNT" diffqec_smoke.sbatch
squeue -u $USER
cat diffqec_smoke-*.out
```

Expected:
- `SMOKE TEST PASSED`
- Test summary from `pytest tests/test_diffqec.py -v`

---

## Step 8: Validate QEC Pipeline Imports

On LUMI (interactive or via a short batch job):

```bash
source $PROJECT_DIR/venv/bin/activate
python - <<'PY'
import stim, qiskit, pymatching, numpy, torch
print("stim:", stim.__version__)
print("qiskit:", qiskit.__version__)
print("pymatching:", pymatching.__version__)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("All imports OK")
PY
```

---

## Step 9: First Production Run (Optional)

After smoke tests pass, submit a real job:

```bash
# Example: run the full DiffQEC smoke with more epochs
sbatch --account="$LUMI_PROJECT_ACCOUNT" diffqec_smoke.sbatch
```

Monitor with `squeue -u $USER`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `module load LUMI/25.09` fails | Check `module avail`; system may have different software stack |
| `cray-python/3.11.7` not found | Try `cray-python/3.11.6` or check `module avail cray-python` |
| Import errors in venv | Re-run `bash setup_lumi_env.sh`; only remove a venv after confirming its path is `/scratch/$USER/venv-junction` |
| Home file-count quota exceeded | Run `bash lumi_deployment/cleanup_lumi.sh --dry-run` first, then without flag; see [storage.md](./storage.md) |
| ROCm/GPU not visible | Confirm `module load partition/G` was run before `module load rocm` |
| Out of budget | Run `lumi-allocations` to check remaining; contact CSC |
| SSH alias unknown | Ask admin; do NOT guess `lumi` vs `lumi2` vs full hostname |

---

## First-Time Summary

```
laptop: cp env.example .env  → fill in LUMI_PROJECT_ACCOUNT/SBATCH_ACCOUNT
lumi:   bash setup_lumi_env.sh
lumi:   source ~/.bashrc    → pick up cache redirects
lumi:   sbatch --account="$LUMI_PROJECT_ACCOUNT" hello_smoke.sbatch  → should print SMOKE OK
lumi:   sbatch --account="$LUMI_PROJECT_ACCOUNT" diffqec_smoke.sbatch → should print SMOKE TEST PASSED
```

Once Steps 1-8 complete, you're ready for [deployment.md](./deployment.md) (day-of workflow). See [storage.md](./storage.md) if you hit quota warnings.