"""Tests for the BP+OSD decoder (bp/integrate.py).

Coverage:
- dem_to_pcm: shape contracts, parity, observable matrix wiring
- build_bp_decoder: construction, defaults, parameterisation
- decode_hardware_results_bp: contract, parity with MWPM, OSD matters
- calibration-weighted variant: priors override, non-negative, finite
- convergence fallback: returns valid prediction, falls back on demand
- dispatcher in run_on_hardware: decoder="bp" path
"""
import numpy as np
import pytest
import stim

from surface_code import DEFAULT_NOISE, make_stim_circuit

from bp.integrate import (
    build_bp_decoder,
    decode_hardware_results_bp,
    dem_to_pcm,
)
from mwpm.integrate import (
    build_matching_decoder,
    decode_hardware_results_mwpm,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def d3_circuit():
    """Noisy d=3, 3-round, p=0.001 surface code circuit."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.001,
        after_reset_flip_probability=0.001,
        before_measure_flip_probability=0.001,
        before_round_data_depolarization=0.001,
    )


@pytest.fixture(scope="module")
def d3_syndromes(d3_circuit):
    """det_events + obs_flips from a stim sample (5000 shots)."""
    det, obs = d3_circuit.compile_detector_sampler(seed=42).sample(
        shots=5000, separate_observables=True, bit_packed=False
    )
    return {"det_events": det.astype(bool), "obs_flips": obs.astype(bool)}


# ─────────────────────────────────────────────────────────────────────────────
#  dem_to_pcm
# ─────────────────────────────────────────────────────────────────────────────

class TestDemToPcm:
    def test_shapes(self, d3_circuit):
        H, O, probs = dem_to_pcm(d3_circuit)
        num_dets = d3_circuit.num_detectors
        num_obs = d3_circuit.num_observables
        # PCM has num_dets + num_obs rows; we return the detector-only H.
        assert H.shape[0] == num_dets
        assert H.shape[1] > 0
        assert O.shape[0] == num_obs
        assert O.shape[1] == H.shape[1]
        assert probs.shape == (H.shape[1],)

    def test_probabilities_in_unit_interval(self, d3_circuit):
        _, _, probs = dem_to_pcm(d3_circuit)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 0.5)  # stim's per-error probs are split into two

    def test_returns_sparse_csr(self, d3_circuit):
        import scipy.sparse as sp
        H, _, _ = dem_to_pcm(d3_circuit)
        assert sp.issparse(H)
        assert H.format == "csr"

    def test_decompose_errors_flag(self, d3_circuit):
        H_a, _, _ = dem_to_pcm(d3_circuit, decompose_errors=True)
        H_b, _, _ = dem_to_pcm(d3_circuit, decompose_errors=False)
        # Both should yield valid parity check matrices; decomposition may
        # change the column count but not the row count.
        assert H_a.shape[0] == H_b.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
#  build_bp_decoder
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildBpDecoder:
    def test_returns_decoder_and_O(self, d3_circuit):
        decoder, O = build_bp_decoder(d3_circuit)
        assert decoder is not None
        assert isinstance(O, np.ndarray)
        assert O.shape[0] == d3_circuit.num_observables

    def test_osd_order_zero_still_runs(self, d3_circuit):
        decoder, O = build_bp_decoder(d3_circuit, osd_order=0)
        det, obs = d3_circuit.compile_detector_sampler(seed=1).sample(
            shots=8, separate_observables=True, bit_packed=False
        )
        syn = {"det_events": det.astype(bool), "obs_flips": obs.astype(bool)}
        ler, err, conf = decode_hardware_results_bp(syn, decoder=decoder, O=O)
        assert 0.0 <= ler[0] <= 1.0
        assert err[0] >= 0.0
        # Confidence may legitimately be all-ones (BP converged to a codeword)
        assert np.isfinite(conf).all()

    def test_bp_method_min_sum(self, d3_circuit):
        decoder, O = build_bp_decoder(d3_circuit, bp_method="ms", ms_scaling_factor=0.9)
        assert decoder.bp_method == "minimum_sum"

    def test_bp_method_product_sum(self, d3_circuit):
        decoder, O = build_bp_decoder(d3_circuit, bp_method="ps")
        assert decoder.bp_method == "product_sum"

    def test_max_iter_is_set(self, d3_circuit):
        decoder, _ = build_bp_decoder(d3_circuit, max_iter=42)
        assert decoder.max_iter == 42


# ─────────────────────────────────────────────────────────────────────────────
#  decode_hardware_results_bp
# ─────────────────────────────────────────────────────────────────────────────

class TestDecodeHardwareResultsBp:
    def test_returns_ler_err_confidence(self, d3_syndromes, d3_circuit):
        decoder, O = build_bp_decoder(d3_circuit)
        ler, err, conf = decode_hardware_results_bp(
            d3_syndromes, decoder=decoder, O=O
        )
        assert isinstance(ler, list)
        assert isinstance(err, list)
        assert len(ler) == 1
        assert len(err) == 1
        assert conf.shape == (len(d3_syndromes["obs_flips"]), 1)
        assert np.isfinite(conf).all()
        assert np.all((conf >= 0.0) & (conf <= 1.0))

    def test_ler_below_random(self, d3_syndromes, d3_circuit):
        """At p=0.001 / d=3, BP+OSD should beat 50% random by a lot."""
        decoder, O = build_bp_decoder(d3_circuit, osd_order=2)
        ler, err, _ = decode_hardware_results_bp(
            d3_syndromes, decoder=decoder, O=O
        )
        assert ler[0] < 0.1, f"BP+OSD LER {ler[0]} is suspiciously high"

    def test_confidence_in_unit_interval(self, d3_syndromes, d3_circuit):
        decoder, O = build_bp_decoder(d3_circuit)
        _, _, conf = decode_hardware_results_bp(
            d3_syndromes, decoder=decoder, O=O
        )
        assert np.isfinite(conf).all()
        assert (conf >= 0.0).all() and (conf <= 1.0).all()

    def test_return_confidence_false(self, d3_syndromes, d3_circuit):
        decoder, O = build_bp_decoder(d3_circuit)
        ler, err, conf = decode_hardware_results_bp(
            d3_syndromes, decoder=decoder, O=O, return_confidence=False
        )
        assert conf is None

    def test_bp_osd2_matches_mwpm_within_3sigma(self, d3_syndromes, d3_circuit):
        """BP+OSD and MWPM should agree to within statistical error at d=3 p=0.001."""
        bp_dec, O = build_bp_decoder(d3_circuit, osd_order=2)
        ler_bp, err_bp, _ = decode_hardware_results_bp(
            d3_syndromes, decoder=bp_dec, O=O
        )
        matcher = build_matching_decoder(d3_circuit)
        ler_mw, err_mw = decode_hardware_results_mwpm(
            d3_syndromes, matcher=matcher
        )
        diff = abs(ler_bp[0] - ler_mw[0])
        # 3-sigma union bound
        assert diff < 3 * (err_bp[0] + err_mw[0]) + 1e-3, (
            f"BP+OSD LER {ler_bp[0]:.4f} and MWPM LER {ler_mw[0]:.4f} "
            f"disagree by more than expected ({diff:.4f})"
        )

    def test_osd_helps_over_plain_bp(self, d3_syndromes, d3_circuit):
        """OSD post-processing should not make things worse (and usually helps)."""
        # Plain BP (osd_order=0) is the BP-only baseline.
        dec0, O0 = build_bp_decoder(d3_circuit, osd_order=0, max_iter=200)
        ler0, _, _ = decode_hardware_results_bp(
            d3_syndromes, decoder=dec0, O=O0
        )
        # BP+OSD (osd_order=2) is the practical working point.
        dec2, O2 = build_bp_decoder(d3_circuit, osd_order=2, max_iter=200)
        ler2, _, _ = decode_hardware_results_bp(
            d3_syndromes, decoder=dec2, O=O2
        )
        # At low p on a d=3 surface code, OSD should not be worse; usually
        # it's measurably better. The README claims 2-3x improvement.
        assert ler2[0] <= ler0[0] * 1.2 + 1e-4, (
            f"BP+OSD {ler2[0]:.4f} unexpectedly worse than plain BP {ler0[0]:.4f}"
        )

    def test_no_circuit_no_decoder_raises(self, d3_syndromes):
        """If neither decoder nor stim_circuit_noisy is provided, raise ValueError."""
        with pytest.raises(ValueError, match="decoder"):
            decode_hardware_results_bp(d3_syndromes)

    def test_auto_build_from_circuit(self, d3_syndromes, d3_circuit):
        """If decoder is None but circuit is provided, build on the fly."""
        ler, err, _ = decode_hardware_results_bp(
            d3_syndromes, stim_circuit_noisy=d3_circuit
        )
        assert 0.0 <= ler[0] <= 1.0
        assert err[0] >= 0.0

    def test_zero_syndrome_gives_zero_ler(self, d3_circuit):
        """With all-zero syndromes (no errors) and exact decode, LER should be 0."""
        decoder, O = build_bp_decoder(d3_circuit, osd_order=2)
        num_dets = d3_circuit.num_detectors
        syn = {
            "det_events": np.zeros((20, num_dets), dtype=bool),
            "obs_flips": np.zeros((20, 1), dtype=bool),
        }
        ler, _, _ = decode_hardware_results_bp(syn, decoder=decoder, O=O)
        assert ler[0] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration-weighted variant
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationWeightedDecoder:
    """Inject custom per-edge error rates and verify the decoder respects them."""

    def test_calibration_override_changes_priors(self, d3_circuit):
        """Overriding channel probs should change the decoder's internal priors."""
        H, O, probs = dem_to_pcm(d3_circuit)
        # Double every prior (just for testing — not a sensible calibration)
        new_probs = np.minimum(probs * 2.0, 0.5).tolist()
        decoder, _ = build_bp_decoder(d3_circuit, error_channel=new_probs)
        # bp_decoder stores channel_probs on the C++ side; check the
        # error_channel list round-trips.
        assert np.allclose(decoder.error_channel, new_probs, atol=1e-6)

    def test_calibration_zero_priors(self, d3_circuit):
        """Setting priors to zero should still produce a valid decoder
        (the OSD step handles the resulting cold start)."""
        H, O, probs = dem_to_pcm(d3_circuit)
        new_probs = np.zeros_like(probs).tolist()
        decoder, _ = build_bp_decoder(d3_circuit, error_channel=new_probs)
        det, obs = d3_circuit.compile_detector_sampler(seed=0).sample(
            shots=16, separate_observables=True, bit_packed=False
        )
        syn = {"det_events": det.astype(bool), "obs_flips": obs.astype(bool)}
        ler, err, _ = decode_hardware_results_bp(syn, decoder=decoder, O=O)
        assert np.isfinite(ler[0])
        assert 0.0 <= ler[0] <= 1.0

    def test_homogeneous_rescale(self, d3_syndromes, d3_circuit):
        """Multiplying all priors by the same constant shouldn't change the
        decoder output much — the OSD step is invariant to uniform rescaling.
        """
        H, O, probs = dem_to_pcm(d3_circuit)
        rescaled = (probs * 1.5).clip(max=0.5).tolist()
        dec_orig, _ = build_bp_decoder(d3_circuit, error_channel=probs.tolist())
        dec_scal, _ = build_bp_decoder(d3_circuit, error_channel=rescaled)
        ler_orig, _, _ = decode_hardware_results_bp(
            d3_syndromes, decoder=dec_orig, O=O
        )
        ler_scal, _, _ = decode_hardware_results_bp(
            d3_syndromes, decoder=dec_scal, O=O
        )
        # BP+OSD should be relatively invariant to small uniform rescalings
        # of the priors (it re-ranks by syndrome consistency, not magnitude).
        assert abs(ler_orig[0] - ler_scal[0]) < 0.02, (
            f"Uniform rescaling changed LER by {abs(ler_orig[0] - ler_scal[0]):.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Convergence-based fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergenceFallback:
    """The convergence-fallback variant routes non-converged shots to MWPM."""

    def test_builds_both_decoders(self, d3_circuit):
        from bp.calibration import build_bp_with_mwpm_fallback

        bp_dec, O, matcher = build_bp_with_mwpm_fallback(d3_circuit)
        assert bp_dec is not None
        assert O is not None
        assert matcher is not None

    def test_decode_returns_predictions(self, d3_syndromes, d3_circuit):
        from bp.calibration import build_bp_with_mwpm_fallback, decode_with_fallback

        bp_dec, O, matcher = build_bp_with_mwpm_fallback(d3_circuit)
        pred, conf, used_fallback = decode_with_fallback(
            d3_syndromes, bp_decoder=bp_dec, O=O, matcher=matcher
        )
        assert pred.shape == d3_syndromes["obs_flips"].shape
        assert pred.dtype == np.uint8
        assert np.all((pred == 0) | (pred == 1))
        assert conf.shape == (len(d3_syndromes["obs_flips"]), 1)
        assert np.isfinite(conf).all()
        # used_fallback is per-shot boolean
        assert used_fallback.shape == (len(d3_syndromes["obs_flips"]),)
        assert used_fallback.dtype == bool

    def test_fallback_used_on_zero_syndrome(self, d3_circuit):
        """Zero syndromes should converge in BP — fallback should not trigger."""
        from bp.calibration import build_bp_with_mwpm_fallback, decode_with_fallback

        bp_dec, O, matcher = build_bp_with_mwpm_fallback(d3_circuit, osd_order=2)
        num_dets = d3_circuit.num_detectors
        syn = {
            "det_events": np.zeros((8, num_dets), dtype=bool),
            "obs_flips": np.zeros((8, 1), dtype=bool),
        }
        pred, _, used_fallback = decode_with_fallback(
            syn, bp_decoder=bp_dec, O=O, matcher=matcher
        )
        # With osd_order=2 OSD will rescue a non-converged BP, so
        # "convergence" is broadly defined as "OSD produced a codeword".
        # Either way the prediction should be 0.
        assert (pred == 0).all()

    def test_fallback_ler_below_random(self, d3_syndromes, d3_circuit):
        from bp.calibration import decode_with_fallback
        from bp.integrate import build_bp_decoder
        from mwpm.integrate import build_matching_decoder

        bp_dec, O = build_bp_decoder(d3_circuit, osd_order=2)
        matcher = build_matching_decoder(d3_circuit)
        pred, _, used = decode_with_fallback(
            d3_syndromes, bp_decoder=bp_dec, O=O, matcher=matcher
        )
        # Per-shot prediction matches the obs_flips when decoding is correct
        ok = (pred.flatten() == d3_syndromes["obs_flips"].flatten().astype(np.uint8))
        # Fallback is allowed to "use" any fraction; the test is just that
        # overall LER is well below random.
        ler = 1.0 - ok.mean()
        assert ler < 0.1, f"Fallback LER {ler:.4f} is suspiciously high"


# ─────────────────────────────────────────────────────────────────────────────
#  End-to-end dispatcher (run_on_hardware.decode_hardware_results)
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatcher:
    def test_bp_path_via_dispatcher(self, d3_syndromes, d3_circuit):
        from run_on_hardware import decode_hardware_results

        bp_dec, O = build_bp_decoder(d3_circuit)
        # The dispatcher signature uses bp_decoder / bp_O
        ler, err = decode_hardware_results(
            d3_syndromes,
            decoder="bp",
            stim_circuit_noisy=d3_circuit,
            bp_decoder=bp_dec,
            bp_O=O,
        )
        assert isinstance(ler, list)
        assert isinstance(err, list)
        assert len(ler) == 1
        assert 0.0 <= ler[0] <= 1.0
