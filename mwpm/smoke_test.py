"""One-shot smoke test for the MWPM decoder.

Usage:
    python -m mwpm.smoke_test

Builds a d=3 noisy surface-code circuit, samples detection events,
runs MWPM, prints LER and a sanity-check that we beat random guessing.
"""
import time
import numpy as np
import stim

from mwpm.integrate import build_matching_decoder, decode_hardware_results_mwpm


def main():
    print("=" * 60)
    print("MWPM Smoke Test")
    print("=" * 60)

    d = 3
    rounds = 3
    shots = 5000
    noise = dict(
        after_clifford_depolarization=0.001,
        after_reset_flip_probability=0.001,
        before_measure_flip_probability=0.001,
        before_round_data_depolarization=0.001,
    )

    # 1. Build circuit + sample.
    print(f"d={d}, rounds={rounds}, shots={shots}, p=0.001")
    t0 = time.time()
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d, rounds=rounds, **noise,
    )
    det_events, obs_flips = circuit.compile_detector_sampler(seed=42).sample(
        shots=shots, separate_observables=True, bit_packed=False
    )
    det_events = det_events.astype(bool)
    obs_flips = obs_flips.astype(bool)
    print(f"  det={det_events.shape}, obs={obs_flips.shape}, "
          f"raw flip rate={obs_flips.mean():.4f}  ({time.time()-t0:.2f}s)")

    # 2. Build matcher.
    t0 = time.time()
    matcher = build_matching_decoder(circuit)
    print(f"  matcher: {matcher.num_detectors} dets, "
          f"{matcher.num_edges} edges  ({time.time()-t0:.2f}s)")

    # 3. Decode.
    t0 = time.time()
    syndromes = {"det_events": det_events, "obs_flips": obs_flips}
    ler, err = decode_hardware_results_mwpm(syndromes, matcher=matcher)
    print(f"  decode: {time.time()-t0:.2f}s")
    print(f"  LER = {ler[0]:.4f} ± {err[0]:.4f}")

    # 4. Random-guess baseline.
    rng = np.random.default_rng(0)
    random_wrong = (rng.random(obs_flips.shape) > 0.5) != obs_flips
    random_ler = random_wrong.mean()
    print(f"  random baseline = {random_ler:.4f}")

    # 5. Acceptance: clearly below 0.4 (random is ~0.5).
    if ler[0] < 0.4:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
