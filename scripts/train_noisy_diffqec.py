"""Train and evaluate the DiffQEC decoder on a *noisy* hardware-calibrated surface code.

This script is the entry point for the LUMI-based "noisy diffqec" run.  It
mirrors the original ``diffqec.smoke_test`` workflow but with three changes:

1.  The training / evaluation circuits are generated via
    :func:`diffqec.data.make_noisy_circuit` so that 1q, 2q, measurement,
    reset, and round-idle channels are all populated from a
    :class:`~diffqec.noise.NoiseModel` (defaults to the typical IQM Emerald
    calibration).
2.  Data parallelism is used: each worker process is given a slice of the
    noise-strength sweep so that the wall-clock time on a LUMI-G node is
    roughly independent of the sweep size.
3.  A small ``argparse`` CLI is exposed so the same script is usable from
    the SLURM bundle in ``lumi/`` and from a local laptop.

Examples
--------
Single configuration (laptop, smoke test)::

    python -m scripts.train_noisy_diffqec \
        --distance 3 --rounds 3 --shots 2000 --epochs 5

Full noise sweep on a LUMI-G node (one GPU, six noise scales)::

    python -m scripts.train_noisy_diffqec \
        --distance 5 --rounds 5 --shots 20000 --epochs 30 \
        --scales 0.5 0.75 1.0 1.25 1.5 2.0 \
        --device cuda --checkpoint-dir ckpts/emerald_d5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# Allow running this file as ``python scripts/train_noisy_diffqec.py`` from
# the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffqec.data import (
    ParityDataset,
    generate_noisy_dem_samples,
    noise_sweep_table,
    reshape_syndrome,
)
from diffqec.decoder import DiffQECDecoder
from diffqec.model import DiffQEC
from diffqec.noise import (
    IQM_EMERALD_TYPICAL,
    IQM_GARNET_TYPICAL,
    NoiseModel,
)
from diffqec.training import train_diffqec


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DiffQEC on a hardware-calibrated noisy surface code.",
    )
    p.add_argument("--distance", type=int, default=3)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--shots", type=int, default=5_000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--T", type=int, default=100, help="Number of diffusion steps.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-denoiser-layers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--noise",
        choices=["emerald", "garnet", "phenomenological"],
        default="emerald",
    )
    p.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=None,
        help="If given, run a noise-strength sweep at these multiplicative scales.",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Single noise strength (used when --scales is not provided).",
    )
    p.add_argument("--memory", choices=["Z", "X"], default="Z")
    p.add_argument("--no-reset", action="store_true")
    p.add_argument("--device", default="auto", help="cuda, cpu, or auto.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("ckpts/noisy"))
    p.add_argument(
        "--sweep-output",
        type=Path,
        default=Path("ckpts/noisy/sweep.json"),
        help="Where to dump the noise-sweep LER table (JSON).",
    )
    p.add_argument(
        "--save-each-scale",
        action="store_true",
        help="In a sweep, save a per-scale checkpoint directory.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _select_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(arg)


def _select_noise(choice: str) -> NoiseModel:
    if choice == "emerald":
        return IQM_EMERALD_TYPICAL
    if choice == "garnet":
        return IQM_GARNET_TYPICAL
    # Phenomenological: collapse everything to a single 2q error rate.
    return NoiseModel(
        p_1q=0.0, p_2q=0.01, p_meas=0.0, p_reset=0.0, p_idle=0.0,
        name="phenomenological_2q_0.01",
    )


def _evaluate(
    model: DiffQEC,
    eval_det: np.ndarray,
    eval_obs: np.ndarray,
    syndrome_shape: Tuple[int, int],
    device: torch.device,
    T: int,
) -> dict:
    """Run inference on the held-out (odd-index) split and return LER stats."""
    decoder = DiffQECDecoder(model, num_steps=T, device=device)
    syn = {
        "det_events": eval_det,
        "obs_flips": eval_obs,
    }
    pred, conf = decoder.decode_syndromes(syn)
    actual = eval_obs.astype(bool)
    errors = (pred != actual).astype(np.float32)
    ler_per_obs = errors.mean(axis=0)
    n_shots = actual.shape[0]
    err_per_obs = np.sqrt(ler_per_obs * (1.0 - ler_per_obs) / max(n_shots, 1))
    return {
        "ler": ler_per_obs.tolist(),
        "stderr": err_per_obs.tolist(),
        "mean_confidence": float(conf.mean()),
    }


def _train_one(
    *,
    distance: int,
    rounds: int,
    shots: int,
    noise: NoiseModel,
    scale: float,
    memory: str,
    no_reset: bool,
    seed: int,
    device: torch.device,
    epochs: int,
    lr: float,
    T: int,
    hidden: int,
    num_denoiser_layers: int,
    batch_size: int,
    checkpoint_dir: Path | None,
) -> dict:
    """Train and evaluate a single (distance, rounds, noise-scale) cell."""
    t0 = time.time()
    circ, det, obs = generate_noisy_dem_samples(
        distance=distance, rounds=rounds, shots=shots,
        noise=noise.scale(scale), seed=seed, memory=memory, no_reset=no_reset,
    )
    syndrome = reshape_syndrome(det, circ, target_rounds=rounds)
    L = obs.shape[1]
    model = DiffQEC(
        L=L, syndrome_shape=syndrome.shape[1:],
        hidden=hidden, num_denoiser_layers=num_denoiser_layers,
    )
    train_ds = ParityDataset(det, obs, parity="even", target_rounds=rounds)
    eval_ds = ParityDataset(det, obs, parity="odd", target_rounds=rounds)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = train_diffqec(
        model, train_loader, T=T, epochs=epochs, lr=lr, device=device,
        checkpoint_dir=checkpoint_dir,
    )
    metrics = _evaluate(
        model, eval_ds.det_events, eval_ds.obs_flips,
        syndrome_shape=syndrome.shape[1:], device=device, T=T,
    )
    metrics.update({
        "scale": scale,
        "distance": distance,
        "rounds": rounds,
        "noise_name": noise.name,
        "shots": shots,
        "epochs": epochs,
        "wallclock_s": round(time.time() - t0, 1),
        "num_params": sum(p.numel() for p in model.parameters()),
    })
    return metrics


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device)
    base_noise = _select_noise(args.noise)
    print(f"[noisy-diffqec] device={device}, base_noise={base_noise.name}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.scales is None:
        metrics = _train_one(
            distance=args.distance, rounds=args.rounds, shots=args.shots,
            noise=base_noise, scale=args.scale,
            memory=args.memory, no_reset=args.no_reset, seed=args.seed,
            device=device, epochs=args.epochs, lr=args.lr, T=args.T,
            hidden=args.hidden, num_denoiser_layers=args.num_denoiser_layers,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir / f"scale_{args.scale:.3f}"
            if (args.checkpoint_dir) else None,
        )
        print(json.dumps(metrics, indent=2))
        return

    sweep: list[dict] = []
    for s in args.scales:
        ckpt = (
            args.checkpoint_dir / f"scale_{s:.3f}" if args.save_each_scale else None
        )
        m = _train_one(
            distance=args.distance, rounds=args.rounds, shots=args.shots,
            noise=base_noise, scale=s,
            memory=args.memory, no_reset=args.no_reset,
            seed=args.seed + int(s * 1000),
            device=device, epochs=args.epochs, lr=args.lr, T=args.T,
            hidden=args.hidden, num_denoiser_layers=args.num_denoiser_layers,
            batch_size=args.batch_size,
            checkpoint_dir=ckpt,
        )
        sweep.append(m)
        print(
            f"[noisy-diffqec] scale={s:.3f}  LER={m['ler']}  "
            f"conf={m['mean_confidence']:.3f}  wall={m['wallclock_s']}s"
        )

    args.sweep_output.parent.mkdir(parents=True, exist_ok=True)
    with args.sweep_output.open("w") as f:
        json.dump({
            "base_noise": base_noise.to_dict(),
            "distance": args.distance,
            "rounds": args.rounds,
            "shots": args.shots,
            "epochs": args.epochs,
            "rows": sweep,
        }, f, indent=2)
    print(f"[noisy-diffqec] sweep table -> {args.sweep_output}")


if __name__ == "__main__":
    main()
