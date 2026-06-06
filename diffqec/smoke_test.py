"""One-shot end-to-end smoke test.

Usage:
    python -m diffqec.smoke_test

Trains a tiny DiffQEC model on synthetic d=3 stim data for ~50 steps
and verifies the full pipeline (data → train → decode) runs without errors.
Reports LER vs random baseline for informational purposes; the acceptance
criterion is that the pipeline completes successfully, not that the model
beats random (a 20-epoch CPU model on 500 shots is not expected to converge
reliably).
"""
import tempfile
from pathlib import Path
import numpy as np
import torch

from diffqec.data import make_test_circuit, generate_dem_samples, reshape_syndrome, ParityDataset
from diffqec.model import DiffQEC
from diffqec.training import train_diffqec
from diffqec.decoder import DiffQECDecoder


def main():
    print("=" * 60)
    print("DiffQEC Smoke Test")
    print("=" * 60)

    # Deterministic seeds for reproducibility in CI
    torch.manual_seed(42)
    np.random.seed(42)

    d = 3
    rounds = 3
    shots = 500
    noise = 0.03

    # 1. Generate synthetic data
    print(f"Generating d={d} rotated surface code, {shots} shots...")
    circuit = make_test_circuit(distance=d, rounds=rounds, noise=noise)
    det_events, obs_flips = generate_dem_samples(circuit, shots, seed=42)
    print(f"  Detectors: {det_events.shape[1]}, Observables: {obs_flips.shape[1]}")

    # Reshape syndromes for model
    syndrome = reshape_syndrome(det_events, circuit, target_rounds=rounds)
    print(f"  Syndrome shape: {syndrome.shape}")

    # 2. Build tiny model
    L = obs_flips.shape[1]
    model = DiffQEC(
        L=L,
        syndrome_shape=syndrome.shape[1:],  # (rounds, D)
        hidden=32,
        num_denoiser_layers=2,
        dropout=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}")

    # 3. Train / eval split (parity)
    train_ds = ParityDataset(det_events, obs_flips, parity="even", target_rounds=rounds)
    eval_ds = ParityDataset(det_events, obs_flips, parity="odd", target_rounds=rounds)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=16, shuffle=False)

    # 4. Train briefly
    device = torch.device("cpu")
    ckpt_dir = Path(tempfile.mkdtemp())
    print(f"Training 20 epochs on CPU (checkpoint dir: {ckpt_dir})...")
    model = train_diffqec(
        model,
        train_loader,
        T=10,
        epochs=20,
        lr=1e-3,
        device=device,
        checkpoint_dir=ckpt_dir,
    )

    # 5. Evaluate
    decoder = DiffQECDecoder(model, num_steps=10, device=device)
    # Use original flat detection events for the pipeline-style dict
    odd_mask = np.arange(len(det_events)) % 2 == 1
    syndromes_eval = {
        "det_events": det_events[odd_mask],
        "obs_flips": obs_flips[odd_mask],
    }
    pred, conf = decoder.decode_syndromes(syndromes_eval)
    actual = syndromes_eval["obs_flips"].astype(bool)

    ler = (pred != actual).mean(axis=0)
    rng = np.random.default_rng(seed=123)
    random_guess = rng.random(actual.shape) > 0.5
    random_ler = (random_guess != actual).mean(axis=0)

    print("-" * 60)
    print("Results (eval set):")
    for i in range(len(ler)):
        print(f"  Observable {i}: LER={ler[i]:.4f}  random={random_ler[i]:.4f}")
    print(f"Mean confidence: {conf.mean():.4f}")
    print("-" * 60)

    # Acceptance: pipeline completes without errors and produces valid output.
    # We do NOT require LER < random — a 20-epoch CPU model on 500 shots
    # is not expected to converge reliably.  Smoke test verifies correctness
    # of the pipeline, not model quality.
    pipeline_ok = (
        np.isfinite(ler).all()
        and np.isfinite(conf).all()
        and pred.shape == actual.shape
    )
    if pipeline_ok:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED — pipeline produced invalid output")
        raise SystemExit(1)


if __name__ == "__main__":
    main()