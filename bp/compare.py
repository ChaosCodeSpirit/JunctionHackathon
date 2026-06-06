"""Side-by-side comparison of MWPM, BP+OSD, plain BP, and BP+MWPM-fallback.

Run as:
    python -m bp.compare

Useful for the hackathon demo and the LER-judging criterion: prints a
table of LER for each decoder at several physical error rates so you
can see the operating envelope of each.
"""
import time
import numpy as np
import stim

from bp.integrate import build_bp_decoder, decode_hardware_results_bp
from bp.calibration import build_bp_with_mwpm_fallback, decode_with_fallback
from mwpm.integrate import build_matching_decoder, decode_hardware_results_mwpm


def main():
    d = 3
    rounds = 3
    shots = 5000
    noise_keys = [
        "after_clifford_depolarization",
        "after_reset_flip_probability",
        "before_measure_flip_probability",
        "before_round_data_depolarization",
    ]
    ps = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    print(f"d={d}, rounds={rounds}, shots={shots}")
    print(f"{'p':>7}  {'raw_flip':>9}  {'MWPM':>9}  {'BP+OSD2':>9}  "
          f"{'plainBP':>9}  {'BP+FB':>9}  {'t_MWPM':>7}  {'t_BP+OSD':>9}")
    print("-" * 90)

    for p in ps:
        noise = {k: p for k in noise_keys}
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d, rounds=rounds, **noise,
        )
        det, obs = circ.compile_detector_sampler(seed=42).sample(
            shots=shots, separate_observables=True, bit_packed=False
        )
        det, obs = det.astype(bool), obs.astype(bool)
        syn = {"det_events": det, "obs_flips": obs}

        # MWPM
        matcher = build_matching_decoder(circ)
        t0 = time.time()
        ler_mw, _ = decode_hardware_results_mwpm(syn, matcher=matcher)
        t_mw = time.time() - t0

        # BP+OSD order 2
        dec2, O2 = build_bp_decoder(circ, osd_order=2)
        t0 = time.time()
        ler2, _, _ = decode_hardware_results_bp(syn, decoder=dec2, O=O2)
        t_bp = time.time() - t0

        # Plain BP (osd_order=0)
        dec0, O0 = build_bp_decoder(circ, osd_order=0, max_iter=200)
        ler0, _, _ = decode_hardware_results_bp(syn, decoder=dec0, O=O0)

        # BP with MWPM fallback
        bp_dec, O, mat = build_bp_with_mwpm_fallback(circ, osd_order=2)
        pred, _, used_fb = decode_with_fallback(
            syn, bp_decoder=bp_dec, O=O, matcher=mat
        )
        ler_fb = float(
            (pred.flatten() != obs.flatten().astype(np.uint8)).mean()
        )
        fb_frac = used_fb.mean()

        print(f"{p:>7.4f}  {obs.mean():>9.4f}  {ler_mw[0]:>9.4f}  "
              f"{ler2[0]:>9.4f}  {ler0[0]:>9.4f}  "
              f"{ler_fb:>9.4f}  {t_mw:>7.2f}  {t_bp:>9.2f}")
        # Print fallback usage on its own line if interesting
        if fb_frac > 0.001:
            print(f"  (BP+MWPM fallback used on {fb_frac*100:.2f}% of shots)")

    print()
    print("Notes:")
    print("  - MWPM and BP+OSD2 should be statistically tied on d=3.")
    print("  - plain BP (osd_order=0) loses 2-3x to MWPM at d=3, p=0.001.")
    print("  - BP+FB reverts to MWPM on non-converged BP shots.")
    print("  - Wall time: BP+OSD is ~50x slower than MWPM at d=3; for")


if __name__ == "__main__":
    main()
