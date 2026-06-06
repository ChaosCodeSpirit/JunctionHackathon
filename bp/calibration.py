"""Calibration-aware BP decoder and MWPM fallback.

Two additions on top of ``bp.integrate``:

1. ``build_bp_with_mwpm_fallback`` — construct a BP+OSD decoder and a
   PyMatching matcher in one call. Both share the same factor graph
   (built from the same stim DEM), so the only difference is the
   decoding algorithm.

2. ``decode_with_fallback`` — run BP+OSD shot by shot; whenever BP
   fails to find a valid codeword (i.e. ``decoder.converge == False``
   and the syndrome is non-trivial), re-decode that shot with MWPM.
   The result is a single per-shot prediction and confidence.

The fallback is most useful in two regimes:

- Model mismatch (calibration is wrong). BP+OSD can be brittle when
  the priors are stale; MWPM, with its hard graph structure, can be
  more robust.
- High-weight syndromes near threshold. BP can fail to converge on
  rare heavy-weight shots; MWPM is exact in that regime.

Both decoders are reused across shots; only the per-shot inner
decode changes.
"""
from typing import Optional, Tuple

import numpy as np
import pymatching
import stim

from bp.integrate import build_bp_decoder
from mwpm.integrate import build_matching_decoder

__all__ = ["build_bp_with_mwpm_fallback", "decode_with_fallback"]


def build_bp_with_mwpm_fallback(
    stim_circuit_noisy: stim.Circuit,
    osd_order: int = 2,
    bp_method: str = "ps",
    schedule: str = "parallel",
    max_iter: int = 30,
) -> Tuple[object, np.ndarray, pymatching.Matching]:
    """Build a BP+OSD decoder paired with a matching MWPM fallback.

    Both decoders operate on the same stim DEM, so they share
    detector/observable indexing. Build once per
    (distance, rounds, noise) and reuse across batches.

    Returns
    -------
    bp_decoder : ldpc.BpOsdDecoder
    O          : np.ndarray   observable matrix from the DEM
    matcher    : pymatching.Matching
    """
    bp_decoder, O = build_bp_decoder(
        stim_circuit_noisy,
        osd_order=osd_order,
        bp_method=bp_method,
        schedule=schedule,
        max_iter=max_iter,
    )
    matcher = build_matching_decoder(stim_circuit_noisy)
    return bp_decoder, O, matcher


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def decode_with_fallback(
    syndromes: dict,
    bp_decoder,
    O: np.ndarray,
    matcher: pymatching.Matching,
    fall_back_on_non_converge: bool = True,
    return_confidence: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run BP+OSD with MWPM fallback on non-converged shots.

    For each shot:

    1. Call ``bp_decoder.decode(syndrome)``.
    2. Check ``bp_decoder.converge``. If False and the syndrome is
       non-trivial, fall back to ``matcher.decode(syndrome)``.
    3. Project the predicted error pattern to the observable space
       and return its soft confidence.

    Parameters
    ----------
    syndromes                 : dict from ``extract_syndromes()``.
    bp_decoder                : pre-built ``ldpc.BpOsdDecoder``.
    O                         : observable matrix from ``build_bp_decoder``.
    matcher                   : pre-built ``pymatching.Matching``.
    fall_back_on_non_converge : if True, route non-converged shots to MWPM.
                                Set False to get pure BP+OSD behaviour while
                                still flagging the non-converged shots.
    return_confidence         : if True, also compute per-shot confidence.

    Returns
    -------
    predicted     : np.ndarray (shots, num_obs)  uint8  hard predictions
    confidence    : np.ndarray (shots, num_obs)  float  in [0, 1]
    used_fallback : np.ndarray (shots,)          bool   True if MWPM
                                                        was used for that shot
    """
    det_events = syndromes["det_events"].astype(np.uint8)
    obs_flips = syndromes["obs_flips"]
    shots = len(det_events)
    num_obs = obs_flips.shape[1]

    predicted = np.zeros((shots, num_obs), dtype=np.uint8)
    confidence = np.zeros((shots, num_obs), dtype=np.float64)
    used_fallback = np.zeros(shots, dtype=bool)

    for i in range(shots):
        s = det_events[i]
        # Step 1: BP+OSD
        e_hat = bp_decoder.decode(s)
        bp_converged = bool(bp_decoder.converge)

        if (not bp_converged) and fall_back_on_non_converge and s.any():
            # Step 2: fall back to MWPM on non-converged, non-trivial shots
            # PyMatching's decode() returns the observable flips directly
            # (length = num_obs), not an error pattern, so we assign
            # directly to predicted[i] and skip the O projection.
            predicted[i] = matcher.decode(s).astype(np.uint8)
            used_fallback[i] = True
        else:
            # BP path: project error pattern through the observable matrix
            predicted[i] = (O.astype(np.uint8) @ e_hat) % 2

        if return_confidence:
            # Soft confidence from BP's LLR. Even if we fell back to MWPM,
            # the LLR from BP is still a useful "uncertainty" signal.
            try:
                llr_e = bp_decoder.log_prob_ratios
                llr_obs = O.astype(np.float64) @ llr_e
                p_obs = _sigmoid(llr_obs)
                confidence[i] = np.where(
                    predicted[i] == 1, p_obs, 1.0 - p_obs
                )
            except Exception:
                # BP didn't run cleanly (e.g. zero syndrome) → neutral
                confidence[i] = 0.5

    return predicted, confidence, used_fallback
