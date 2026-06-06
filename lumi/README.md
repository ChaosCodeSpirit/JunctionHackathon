# Running the noisy DiffQEC pipeline on LUMI

This directory contains everything you need to run the noisy extension of
`diffqec` on the [LUMI](https://www.lumi-supercomputer.eu/) supercomputer
hosted by CSC in Finland.

## Why LUMI

* **LUMI-G**: 2,928 AMD MI250X GPUs (each with two GCDs = 128 GB HBM2e).
  The diffusion decoder is a small, dense model and fits comfortably on a
  single GCD, so one LUMI-G node (4 GCDs) can train four decoder instances
  in parallel via SLURM's job-array feature.
* **LUMI-C**: 64-core AMD EPYC CPUs.  The DEM data generation (Stim) is
  CPU-bound and embarrassingly parallel — `noise_sweep_table` already
  partitions the work by noise scale.
* **Storage**: `/scratch/<project>` is a parallel Lustre filesystem that
  can sustain the multi-GB write rate the sweep produces.

## Files

| File             | Purpose                                                    |
|------------------|------------------------------------------------------------|
| `install.sh`     | One-shot environment setup on a LUMI login node            |
| `train.slurm`    | Single-cell training job (1 node, 1 GCD)                   |
| `sweep.slurm`    | Array job: 2 distances × 6 noise scales = 12 GPU jobs      |
| `README.md`      | This file                                                  |

## Quick start

```bash
# 1.  Log in to LUMI and clone the repo on /scratch
ssh lumi.csc.fi
cd /scratch/<project>/$USER
git clone https://github.com/<your-org>/JunctionHackathon.git
cd JunctionHackathon
git checkout feature/diffqec-noisy

# 2.  Install the stack on the login node (uses the ROCm PyTorch wheel)
module load LUMI/24.03 partition/G
bash lumi/install.sh
source .venv-lumi/bin/activate

# 3.  Submit a single training job
sbatch --account=<project_xxx> lumi/train.slurm

# 4.  Or submit the full noise sweep
sbatch --account=<project_xxx> lumi/sweep.slurm
```

## Customising the runs

Both SLURM scripts read their configuration from environment variables, so
the typical workflow is:

```bash
# A larger, longer d=7 run
sbatch --account=<project_xxx> \
    --export=DISTANCE=7,ROUNDS=7,SHOTS=40000,EPOCHS=50 \
    lumi/train.slurm

# A 3-distance, 8-scale sweep = 24 jobs
sbatch --account=<project_xxx> \
    --export=DISTANCES_CSV="3 5 7",SCALES_CSV="0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0" \
    lumi/sweep.slurm
```

The default `partition=small-g` is meant for short jobs (< 6h) and is the
easiest queue to get into.  For longer runs use `partition=standard-g` (you
need to be in the appropriate LUMI user group) and bump `--time=` to e.g.
`06:00:00`.

## Outputs

The sweep writes one row per `(distance, scale)` cell into
`ckpts/sweep-<jobid>/sweep.json`.  The format is:

```json
{
  "base_noise": { "p_1q": 0.001, "p_2q": 0.005, ... },
  "distance": 5,
  "rounds": 5,
  "shots": 15000,
  "epochs": 20,
  "rows": [
    {
      "scale": 1.0,
      "ler": [0.0241],
      "stderr": [0.0012],
      "mean_confidence": 0.81,
      "wallclock_s": 412.7,
      "num_params": 107329
    },
    ...
  ]
}
```

A minimal plotting recipe (in the same Python env) is:

```python
import json, matplotlib.pyplot as plt
d = json.load(open("ckpts/sweep-12345/sweep.json"))
scales = [r["scale"] for r in d["rows"]]
ler    = [r["ler"][0] for r in d["rows"]]
err    = [r["stderr"][0] for r in d["rows"]]
plt.errorbar(scales, ler, yerr=err, marker="o")
plt.xlabel("noise scale"); plt.ylabel("LER")
plt.title(f"d={d['distance']}, r={d['rounds']}, noise={d['base_noise']['name']}")
plt.yscale("log"); plt.grid(True); plt.show()
```

## Notes for the LUMI-G ROCm stack

* PyTorch on LUMI-G uses ROCm 6.0.  We pin `torch>=2.4,<2.6` because the
  upstream ROCm 6.0 wheel is built against that ABI.
* One MI250X is reported as two GCDs to PyTorch (`cuda:0`, `cuda:1`).  We
  use one GCD per task so a 4-GCD node can host four array tasks
  simultaneously.
* `TORCH_HOME` and `HF_HOME` are redirected to `/scratch` to keep the
  login node's home directory from filling up.
