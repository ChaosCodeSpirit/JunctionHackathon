"""One-shot smoke test for the BP decoder.

Usage:
    python -m bp.smoke_test

Builds a d=3 noisy surface-code circuit, samples detection events,
runs BP-OSD, prints LER and a side-by-side with MWPM. Mirrors
``mwpm.smoke_test`` in shape.
"""
import time
import numpy as np
import stim

from bp.integrate import build_bp_decoder, decode_hardware_results_bp


def main():
    print("=" * 60)
    print("BP-OSD Smoke Test")
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

    # 2. Build BP-OSD decoder.
    t0 = time.time()
    decoder, O = build_bp_decoder(circuit, max_iter=30, schedule="parallel",
                                    osd_order=2)
    print(f"  BP-OSD decoder built  ({time.time()-t0:.2f}s)")

    # 3. Decode.
    t0 = time.time()
    syndromes = {"det_events": det_events, "obs_flips": obs_flips}
    ler, err, confidence = decode_hardware_results_bp(syndromes, decoder=decoder, O=O)
    bp_time = time.time() - t0
    print(f"  BP-OSD decode: {bp_time:.2f}s")
    print(f"  BP-OSD LER = {ler[0]:.4f} ± {err[0]:.4f}")
    print(f"  confidence: min={confidence.min():.3f}, "
          f"mean={confidence.mean():.3f}, max={confidence.max():.3f}")

    # 4. Cross-check with MWPM.
    from mwpm.integrate import build_matching_decoder, decode_hardware_results_mwpm
    t0 = time.time()
    matcher = build_matching_decoder(circuit)
    mwpm_ler, mwpm_err = decode_hardware_results_mwpm(syndromes, matcher=matcher)
    print(f"  MWPM decode: {time.time()-t0:.2f}s")
    print(f"  MWPM  LER = {mwpm_ler[0]:.4f} ± {mwpm_err[0]:.4f}")

    # 5. Random-guess baseline.
    rng = np.random.default_rng(0)
    random_wrong = (rng.random(obs_flips.shape) > 0.5) != obs_flips
    random_ler = random_wrong.mean()
    print(f"  random baseline = {random_ler:.4f}")

    # 6. Acceptance: clearly below 0.4 (random is ~0.5).
    if ler[0] < 0.4:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

