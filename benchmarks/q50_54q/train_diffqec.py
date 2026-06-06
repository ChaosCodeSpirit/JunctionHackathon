"""Train a small DiffQEC model on the 54-qubit surface-code syndrome data.

For the LUMI 54-qubit benchmark, we want to compare:
  - "no decoding" (trivial baseline — predict 0, LER ≈ 0.03 at p=1e-4)
  - DiffQEC (neural decoder)

This script generates training data on the fly via the Qiskit-Aer stabilizer
simulator, trains a tiny DiffQEC for a few epochs, and saves a checkpoint
that `benchmark.py` can load.

Usage
-----
    python -m benchmarks.q50_54q.train_diffqec --p 0.001 --shots 4096 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .circuit import SurfaceCode54Q
from .simulate import run_stabilizer_simulation


def syndromes_to_x0(syn: dict) -> np.ndarray:
    """The 'clean' logical state to denoise towards is the actual observable.

    For a memory-Z experiment the decoder target is the data parity (XOR of
    all data Z measurements) — i.e. the obs_flips array.  We use this as
    the 'x0' (clean logical correction) for diffusion training.
    """
    return syn["obs_flips"].astype(np.float32)


def reshape_syndrome_for_model(syn: dict, n_rounds: int = 1) -> np.ndarray:
    """Reshape (shots, n_det) → (shots, n_rounds, n_det//n_rounds)."""
    det = syn["det_events"].astype(np.float32)
    n_det = det.shape[1]
    if n_det % n_rounds != 0:
        # pad to multiple of n_rounds
        pad = n_rounds - (n_det % n_rounds)
        det = np.concatenate([det, np.zeros((det.shape[0], pad), dtype=np.float32)], axis=1)
        n_det = det.shape[1]
    per_round = n_det // n_rounds
    return det.reshape(det.shape[0], n_rounds, per_round)


def train_diffqec(
    p: float = 0.001,
    shots: int = 4096,
    epochs: int = 10,
    batch_size: int = 64,
    hidden: int = 64,
    n_layers: int = 3,
    lr: float = 1e-3,
    n_rounds: int = 1,
    T: int = 50,
    seed: int = 42,
    device: str = "cpu",
    out_path: str = "diffqec_54q.pt",
) -> dict:
    """Train DiffQEC and save checkpoint."""
    from diffqec.model import DiffQEC
    from diffqec.diffusion import (
        cosine_schedule,
        sample_xt,
        get_x0_from_logits,
    )

    def ce_loss(logits, x0):
        # logits: (B, L, 2), x0: (B, L) float {0,1}
        log_probs = torch.log_softmax(logits, dim=-1)
        target = x0.long()
        return torch.nn.functional.nll_loss(
            log_probs.view(-1, 2), target.view(-1), reduction="mean"
        )

    spec = SurfaceCode54Q(distance=5, rounds=n_rounds, p=p)
    syn = run_stabilizer_simulation(spec, shots=shots, seed=seed)
    x0 = syndromes_to_x0(syn)                       # (shots, 1) — the logical
    syn_in = reshape_syndrome_for_model(syn, n_rounds)  # (shots, R, D)

    # Dataset
    x0_t = torch.from_numpy(x0).to(device)
    syn_t = torch.from_numpy(syn_in).to(device)

    L = x0.shape[1]
    syndrome_shape = (n_rounds, syn_in.shape[2])

    model = DiffQEC(
        L=L,
        syndrome_shape=syndrome_shape,
        hidden=hidden,
        num_denoiser_layers=n_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = cosine_schedule(T)

    n = x0_t.size(0)
    history = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        # shuffle
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x0b = x0_t[idx]
            synb = syn_t[idx]
            t = torch.randint(1, T + 1, (x0b.size(0),), device=device)
            xt = sample_xt(x0b, t, schedule[0])  # noisy
            logits = model(xt, t, synb)
            loss = ce_loss(logits, x0b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        history.append(avg)
        print(f"  epoch {ep+1:>3d}/{epochs}  loss={avg:.4f}")

    elapsed = time.perf_counter() - t0

    # Quick eval: argmax accuracy on the train set (over-fit, but tells us
    # if the model is learning)
    with torch.no_grad():
        logits = model(x0_t, torch.ones(n, device=device, dtype=torch.long), syn_t)
        pred = get_x0_from_logits(logits).cpu().numpy()
        train_acc = float((pred == x0).mean())

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "L": L,
        "syndrome_shape": syndrome_shape,
        "hidden": hidden,
        "n_layers": n_layers,
        "T": T,
        "p_train": p,
        "history": history,
        "train_acc": train_acc,
        "elapsed_s": elapsed,
    }, out_path)
    print(f"\nSaved {out_path}  (train_acc={train_acc:.4f}, elapsed={elapsed:.1f}s)")
    return {
        "train_acc": train_acc,
        "history": history,
        "elapsed_s": elapsed,
        "model_path": out_path,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p", type=float, default=0.001)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_rounds", type=int, default=1)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--out", type=str, default="diffqec_54q.pt")
    args = p.parse_args()

    train_diffqec(
        p=args.p,
        shots=args.shots,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        n_layers=args.n_layers,
        lr=args.lr,
        n_rounds=args.n_rounds,
        T=args.T,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
