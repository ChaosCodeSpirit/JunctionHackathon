# LUMI Storage — Quotas, Warnings, and Cleanup

> You got here because LUMI warned you about **home directory file-count quota**
> or **storage billing units (SBU)**.  This page explains both, tells you how to
> fix them right now, and shows you how to prevent them from recurring.

---

## LUMI Storage Tiers at a Glance

| Tier | Path | Quota (LUMI-G) | Quota (LUMI-C) | Billed? | Auto-purge? |
|---|---|---|---|---|---|
| **Home** | `/users/<u>` | 50 GB, **100 000 files** | 512 GB, 1 M files | No | No |
| **Scratch (per-user)** | `/scratch/<u>` | No hard cap* | No hard cap* | **No SBU** | 90 days idle |
| **Scratch (project)** | `$SCRATCH` → `/scratch/<project>` | No hard cap* | No hard cap* | **No SBU** | 90 days idle |
| **Project** | `/project/<project>` | 1 TB default | 1 TB default | **Yes — SBU** (TB·day) | No |

\* Soft quotas exist; performance degrades badly above a few million files on Lustre.

---

## The Two Warnings You Will See

### 1. "home directory file count quota exceeded"

**What it means.** Your `/users/<u>` has more than 100 000 files (LUMI-G) or
1 M files (LUMI-C).  Login becomes slow; new file creation may fail silently.

**Why it hits QEC projects specifically.** The JunctionHackathon stack
(Qiskit + Stim + PyMatching + torch + JAX) brings in ~30–50 k small files
just in a Python venv.  Pip, uv, matplotlib, Jupyter, and HuggingFace caches
add another 10–30 k.  Add `__pycache__/`, `.pytest_cache/`, shot-result files,
and you are over the limit before you've run a single job.

**Immediate fix.** Run the cleanup script with `--dry-run` first:

```bash
# On LUMI — always dry-run first
bash lumi_deployment/cleanup_lumi.sh --dry-run

# If the dry-run output looks reasonable:
bash lumi_deployment/cleanup_lumi.sh
```

If the default run is not enough (still over quota), re-run with `--deep`:

```bash
bash lumi_deployment/cleanup_lumi.sh --deep
```

`--deep` removes `~/.cache` entirely.  Everything in it will regenerate on
next use, but you pay the time cost of re-downloading wheels.

**Prevention.** Running `setup_lumi_env.sh` (or `cleanup_lumi.sh` with default
flags) writes an idempotent block to `~/.bashrc` that re-points all transient
caches (`$XDG_CACHE_HOME`, `$PIP_CACHE_DIR`, etc.) to `/scratch/$USER/cache`.
This means from your next shell onward, pip/uv/huggingface/jupyter caches go
directly to scratch, not home — the file-count problem cannot recur.

### 2. "storage billing unit" / project quota warning

**What it means.** Your CSC project (`project_46…`) is consuming storage on
`/project/<id>`, which is billed in **SBU** (storage billing units, measured in
TB·day).  The LUMI allocation comes with a fixed SBU budget; if your project
stores data that sits idle, SBU consumption accumulates.

**Why it happens.** Most common causes:

- Large shot dumps, decoder checkpoints, or simulation outputs left on
  `/project/` instead of `/scratch/`.
- `rsync` or `scp` putting data into `/project/` by default.
- Stale result directories from previous hackathon days.

**Immediate fix.** Check what's consuming SBU:

```bash
csc-workspaces                          # list all project workspaces + SBU
du -sh /project/$SLURM_JOB_ACCOUNT/*    # find biggest directories
```

Then either:

- Move large, active data to `/scratch/$USER/` or `/scratch/$SLURM_JOB_ACCOUNT/`
  (no SBU, but auto-purged after 90 days idle).
- Delete data you no longer need.
- Archive final results to your laptop via `rsync -avP lumi:… ./results/`.

---

## Where Things Should Live

| What | Put it on | Why |
|---|---|---|
| Source code, notebooks, `.git` | `$HOME` or `$PROJECT` | Small; needs permanence |
| Python venv | `/scratch/$USER/venv-junction` | 30–50 k files; blows home quota |
| Pip/uv/hf/jupyter caches | `/scratch/$USER/cache` | Transient; we re-point via `~/.bashrc` |
| `__pycache__/`, `.pytest_cache/` | (deleted; regenerated) | Debris; `cleanup_lumi.sh` removes |
| Active simulation data, shot dumps | `/scratch/$USER/` or `/scratch/<project>` | Large, temporary, no SBU |
| Decoder checkpoints, intermediate results | `/scratch/$USER/` | Large, temporary |
| Final results to keep past 90 days | `/project/<project>` | Billed in SBU; only keep what you need |
| SBATCH job output (`*.out`, `*.err`) | `/scratch/<project>` or `$HOME` (small) | Small per-file; many jobs add up |

---

## Day-to-Day Habits That Prevent Quota Issues

1. **Always `source ~/.bashrc` after first run of `setup_lumi_env.sh`.**
   The cache-redirect block takes effect in new shells only.

2. **Never build a venv in `$HOME`.**
   `setup_lumi_env.sh` now puts it on `/scratch/$USER/venv-junction` and
   symlinks it into the project.  If you `python -m venv .venv` manually,
   build it on scratch too:

   ```bash
   python -m venv /scratch/$USER/venv-custom
   ln -s /scratch/$USER/venv-custom .venv
   ```

3. **Clone shallow when you can.**

   ```bash
   git clone --depth 1 <url>   # saves ~10k files in .git/objects
   ```

4. **Don't write per-shot files.**  If your pipeline dumps one file per syndrome
   round, batch them into `.npy` or `.npz` files instead.  A single 10 MB `.npz`
   counts as one file; 10 000 one-line `.txt` files count as 10 000 files and
   destroy Lustre performance.

5. **Clean up after debugging.**  `__pycache__/` and `.pytest_cache/` regenerate
   automatically.  Run `cleanup_lumi.sh` periodically or before big job
   submissions.

6. **Move results off `/project/` when done.**  rsync to your laptop, then
   delete from LUMI.  Sitting data costs SBU.

---

## The Cleanup Script — What It Does

| Flag | What it removes | Risk |
|---|---|---|
| *(default)* | `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.ipynb_checkpoints/`, `*.pyc`, `.coverage`, `~/.cache/{pip,uv,huggingface,matplotlib,jupyter,torch,triton,pytest,Cython}`, `~/.ipython`, `~/.local/share/jupyter/runtime/*`, `~/.jupyter/runtime/*`, `~/.config/{pip,uv}` | None — all auto-regenerated |
| `--deep` | All of the above **plus** `~/.cache/` entirely | Low — caches rebuild on next use, but downloads take time |
| `--no-repoint` | Skip the `~/.bashrc` cache-redirect write | Use this if you manage your env vars yourself |
| `--dry-run` | Print what would be deleted; change nothing | Safe — always run this first |

The script is **idempotent**: running it multiple times is safe.  It uses
`find -xdev` to stay on the home filesystem and never crosses into `/scratch`
or `/project`.

---

## Quick Reference

```bash
# Diagnose — run on LUMI
lfs quota -u $USER ~                  # space + file-count quota
find ~ -xdev -type f | wc -l          # current file count
du -sh ~/.cache ~/.local ~/* 2>/dev/null | sort -h | tail    # biggest dirs
csc-workspaces                         # SBU usage

# Clean up (always dry-run first)
bash lumi_deployment/cleanup_lumi.sh --dry-run
bash lumi_deployment/cleanup_lumi.sh

# If still over quota
bash lumi_deployment/cleanup_lumi.sh --deep

# One-time setup (venv on scratch + cache redirects)
bash lumi_deployment/setup_lumi_env.sh
source ~/.bashrc   # pick up cache redirects

# Verify after cleanup
lfs quota -u $USER ~
find ~ -xdev -type f | wc -l
```

---

## Related Docs

- [preflight.md](./preflight.md) — first-time LUMI setup (~30 min)
- [deployment.md](./deployment.md) — day-of job submission workflow
- [README.md](./README.md) — LUMI docs index + quick reference

## Scripts

| Script | Location | What it does |
|---|---|---|
| `cleanup_lumi.sh` | `lumi_deployment/` | Remove Python debris + caches; re-point caches to scratch |
| `setup_lumi_env.sh` | `lumi_deployment/` | Bootstrap venv on scratch; set up cache redirects |