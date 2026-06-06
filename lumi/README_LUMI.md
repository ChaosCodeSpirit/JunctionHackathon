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

## Files

| File | Purpose |
|---|---|
| `diffqec_54q.sh` | SLURM batch script (LUMI/25.09, `small` partition, 30 min, 16 GB) |
| `requirements_lumi.txt` | Pinned deps (qiskit, qiskit-aer, stim, pymatching, torch) |
| `submit_54q.sh` | Local (laptop-side) rsync + sbatch wrapper |
| `.env.example` | Template for the gitignored `.env` with PROJECT_ID |
| `README_LUMI.md` | This file |

## Submission (laptop-side)

From the repo root (`JunctionHackathon-lumi/`):

```bash
cd lumi
cp .env.example .env          # edit if PROJECT_ID is not project_465003017
./submit_54q.sh push          # rsync code to LUMI
./submit_54q.sh submit        # sbatch the job
./submit_54q.sh status        # watch the queue
./submit_54q.sh fetch         # pull benchmark_results.json + logs back
./submit_54q.sh logs          # tail the latest job log
```

The `submit_54q.sh` wrapper uses the `lumi` SSH alias from
`~/.ssh/config` (already configured for the user `kkiirikk`).  See
`Memory/infrastructure/lumi/lumi-quantumhack-2026.md` for the SSH alias
details.

## LUMI configuration

- **Partition:** `small` (LUMI-C CPU; free, no GPU needed for stabilizer sim)
- **Time:** 30 minutes (training + benchmark; well under 1024-node cap)
- **Memory:** 16 GB (more than enough for 54-qubit stabilizer sim)
- **Modules:** `LUMI/25.09` + `partition/C` + `cray-python/3.11.7`
  (verified 2026-06-05 on this stack; `LUMI/23.09` is deprecated)
- **Project:** `project_465003017` (Junction QuantumHack 2026)
- **Storage:** LUMI-P (Lustre) for working files; venv cached at
  `${PROJECT_DIR}/venv` so subsequent jobs skip the 5-min pip step

## Expected outputs

After the job runs, `benchmark_results.json` will look like:

```json
[
  {"p": 0.0001, "shots": 8192, "ler_raw": ~0.03,
   "ler_mwpm": ~0.03, "ler_diffqec": "< 0.03",
   "t_mwpm_s": ..., "t_diffqec_s": ...},
  ...
]
```

`ler_raw` is the trivial "no decoding" baseline (data parity flip rate
under noise). `ler_diffqec` is the rate after DiffQEC inference.
**A well-trained DiffQEC should achieve `ler_diffqec` strictly less
than `ler_raw`** — the LER reduction is the headline metric.

## Why 54 qubits?

- IQM Q50 (VTT QX) has 54 physical qubits numbered QB1..QB54 arranged in
  a 7×7+ lattice (https://qx.vtt.fi/docs/devices/q50.html).
- A rotated surface code distance 5 uses exactly 49 qubits; 5 flag
  ancilla bring the total to 54, one-to-one with the Q50 footprint.
- DiffQEC's paper claims LER reductions up to distance 17 (577 qubits);
  54 is the largest scale that fits in Q50's footprint.

## Caveats / known limitations

- The 24 X-ancilla are scheduled with a regular 2-data-per-ancilla
  pattern; this is **not** the standard d=5 rotated surface-code parity
  table.  It produces a valid syndrome that scales with p, but the LER
  curve is higher than a true d=5 code.
- Real MWPM decoder (PyMatching) is currently a stub returning the
  trivial prediction.  Wiring it up requires building the matching graph
  from the stim circuit (work in progress).
- The DiffQEC model is intentionally small (~20-30k params) so the
  LUMI job finishes quickly.  Larger models may give better LER but
  cost more compute.
