# 54-qubit DiffQEC benchmark on LUMI

This folder contains the LUMI submission artifacts for benchmarking the
**DiffQEC** neural decoder (arxiv 2604.24640) on a **54-qubit surface
code** that mirrors the IQM Q50 / VTT QX physical qubit count.

## What this benchmark does

- Builds a 54-qubit memory-Z experiment:
  rotated surface code distance 5 (25 data + 24 X-ancilla = 49 qubits) + 5
  flag ancilla = **54 qubits**, matching Q50's QB1..QB54 layout.
- Simulates it on **Qiskit-Aer stabilizer simulator** with symmetric
  depolarizing noise (1Q: p, 2Q: 10p, measurement: p, reset: p).
  Statevector sim is infeasible at 54 qubits (2^54 ≈ 16 PB RAM); the
  stabilizer method is the natural choice for QEC.
- Trains a small DiffQEC (torch) on the syndromes, then evaluates it
  against the trivial "no decoding" baseline across p ∈ {1e-4, 3e-4,
  1e-3, 3e-3, 1e-2}.

## Real LUMI run (job 19078351, 2026-06-06)

| p       | LER_raw (no decoding) | LER_diffqec (trained) | t_diffqec (s) | shots |
|--------:|----------------------:|----------------------:|--------------:|------:|
| 0.0001  | 0.0282                | **0.0000**            | 0.084         | 8192  |
| 0.0003  | 0.0762                | **0.0000**            | 0.056         | 8192  |
| 0.001   | 0.2061                | **0.0000**            | 0.050         | 8192  |
| 0.003   | 0.4084                | **0.0000**            | 0.049         | 8192  |
| 0.01    | 0.5028                | **0.0000**            | 0.051         | 8192  |

- **Host:** `nid002591` (LUMI-C small partition)
- **Wall time:** ~5 min total (42 s training, ~4 min benchmarking across 5 p values)
- **Module stack:** `LUMI/25.09` + `partition/C` + `cray-python/3.11.7` + `torch 2.12.0+cu130`
- **Project:** `project_465003017` (Junction QuantumHack 2026)

**Honest interpretation:** `ler_diffqec = 0` reflects training-set
memorization (the model has 1 logical bit to predict and converges to
train_acc = 1.0 on p=0.001 data; the inference path uses `t=1, x0=obs_flips`
which collapses the diffusion to a one-shot prediction). On **out-of-distribution**
p (e.g. p=0.0001 vs train p=0.001) the model still perfectly recovers
because the (syndrome, parity) relationship is consistent across p. A
proper out-of-distribution holdout + multi-p training is the next step.

Full log: `lumi/logs/run-19078351.log`
JSON results: `lumi/benchmark_results_lumi.json` (also at
`/scratch/project_465003017/kkiirikk/junctionhackathon-lumi/benchmark_results.json` on LUMI)

## Files

| File | Purpose |
|---|---|
| `diffqec_54q.sh` | SLURM batch script (LUMI/25.09, `small` partition, 30 min, 16 GB) |
| `requirements_lumi.txt` | Pinned deps (qiskit, qiskit-aer, stim, pymatching, torch) |
| `submit_54q.sh` | Local (laptop-side) rsync + sbatch wrapper |
| `.env.example` | Template for the gitignored `.env` with PROJECT_ID |
| `logs/run-19078351.log` | LUMI log from the real run |
| `benchmark_results_lumi.json` | JSON results from the real run |
| `README_LUMI.md` | This file |

## Submission (laptop-side)

From the repo root (`JunctionHackathon-lumi/`):

```powershell
# Rsync the repo to LUMI (one time per change)
tar --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache' `
    --exclude='venv' --exclude='.ruff_cache' --exclude='*.pt' `
    --exclude='uv.lock' -czf - . `
  | ssh lumi "mkdir -p ~/junctionhackathon-lumi && tar -xzf - -C ~/junctionhackathon-lumi"

# Submit the job
ssh lumi "cd ~/junctionhackathon-lumi && sbatch lumi/diffqec_54q.sh"

# Tail the log
ssh lumi "cat ~/junctionhackathon-lumi/diffqec_54q_<JOBID>.log"

# Pull results
ssh lumi "cat /scratch/project_465003017/kkiirikk/junctionhackathon-lumi/benchmark_results.json" `
  > lumi/benchmark_results_lumi.json
```

`tar` is used because `rsync` is not in the default Windows OpenSSH
install. The same `submit_54q.sh` wrapper handles this transparently
on Linux/macOS where rsync is available.

The submit script uses the `lumi` SSH alias from `~/.ssh/config`
(already configured for the user `kkiirikk`).

## LUMI configuration

- **Partition:** `small` (LUMI-C CPU; stabilizer sim doesn't need GPU)
- **Time:** 30 minutes (training + benchmark; well under 1024-node cap)
- **Memory:** 16 GB
- **Modules:** `LUMI/25.09` + `partition/C` + `cray-python/3.11.7`
  (verified 2026-06-05; the older `LUMI/23.09` is deprecated)
- **Project:** `project_465003017` (Junction QuantumHack 2026)
- **Storage:** `WORKDIR=/scratch/project_465003017/$USER/junctionhackathon-lumi`
  on LUMI-P (Lustre). `$HOME` has a 1 GB quota, too small for torch.

## Caveats / known limitations

- The 24 X-ancilla are scheduled with a regular 2-data-per-ancilla
  pattern; this is **not** the standard d=5 rotated surface-code parity
  table.  It produces a valid syndrome that scales with p, but the LER
  curve is higher than a true d=5 code.
- Real MWPM decoder (PyMatching) is currently a stub returning the
  trivial prediction.  Wiring it up requires building the matching graph
  from the stim circuit (work in progress).
- The DiffQEC model is intentionally small (~20-30k params) so the
  LUMI job finishes quickly.  The current `ler_diffqec=0` is training-
  set memorization, not a true LER reduction.
- For a meaningful LER-reduction claim, the model needs:
  (a) training across multiple p values,
  (b) a held-out test set, and
  (c) proper iterative diffusion sampling (T > 1 steps).
