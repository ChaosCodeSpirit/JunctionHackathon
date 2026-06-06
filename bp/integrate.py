"""Belief-propagation decoder for the rotated surface code.

Pipeline glue. The decoder has two parts:

    1. build_bp_decoder(circuit)         # stim DEM -> parity check + priors
    2. decoder.decode(syndrome) -> (error_vector, LLRs)  # batched via ldpc

By default this uses BP+OSD (ordered statistics decoding) with
``osd_order=2``, which is the practical working point for QEC: at
d=3, p=0.001 it matches MWPM exactly. Vanilla loopy BP alone
converges to a stable but sub-optimal fixed point on surface codes
because of short cycles in the factor graph; OSD post-processes the
BP result to recover the optimal solution. Set ``osd_order=0`` to
disable OSD and run plain BP.
"""
from typing import List, Optional, Sequence, Tuple
import numpy as np
import scipy.sparse as sp
import stim
import ldpc

__all__ = ["build_bp_decoder", "decode_hardware_results_bp", "dem_to_pcm"]


# ─────────────────────────────────────────────────────────────────────────────
#  DEM -> parity check matrix + channel probabilities
# ─────────────────────────────────────────────────────────────────────────────

def dem_to_pcm(
    stim_circuit_noisy: stim.Circuit,
    decompose_errors: bool = True,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Convert a noisy Stim circuit's DEM into a BP-ready factor graph.

    The BP decoder needs a binary parity check matrix ``H`` of shape
    ``(num_detectors, num_errors)`` where ``H[d, e] = 1`` iff error
    mechanism ``e`` toggles detector ``d``. Each error mechanism also
    has a probability (the channel prior). The matrix ``O`` of shape
    ``(num_observables, num_errors)`` gives the observable flips.

    Returns
    -------
    H      : scipy.sparse.csr_matrix  (num_detectors, num_errors)
    O      : np.ndarray               (num_observables, num_errors) uint8
    probs  : np.ndarray               (num_errors,) float64
    """
    dem = stim_circuit_noisy.detector_error_model(decompose_errors=decompose_errors)
    num_dets = dem.num_detectors
    num_obs = dem.num_observables

    # First pass: collect (prob, det_indices, obs_indices) per error instruction.
    errors: List[Tuple[float, List[int], List[int]]] = []
    for instr in dem:
        if instr.type != "error":
            continue
        prob = instr.args_copy()[0]
        dets: List[int] = []
        obs: List[int] = []
        for t in instr.targets_copy():
            if t.is_separator():
                continue
            if t.is_relative_detector_id():
                # In stim's DEM, each relative_detector_id target's `val`
                # IS the absolute detector id. (The "relative" refers to
                # the implicit -1 starting sentinel, not to the previous
                # target's id.)
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        errors.append((prob, dets, obs))

    # Second pass: build the parity check matrix and observable matrix.
    num_errs = len(errors)
    pcm = np.zeros((num_dets + num_obs, num_errs), dtype=np.uint8)
    probs = np.zeros(num_errs, dtype=np.float64)
    for e_idx, (p, dets, obs) in enumerate(errors):
        probs[e_idx] = p
        for d in dets:
            pcm[d, e_idx] ^= 1
        for o in obs:
            pcm[num_dets + o, e_idx] ^= 1

    H = sp.csr_matrix(pcm[:num_dets, :])
    O = pcm[num_dets:, :]
    return H, O, probs


# ─────────────────────────────────────────────────────────────────────────────
#  Build + decode
# ─────────────────────────────────────────────────────────────────────────────

def build_bp_decoder(
    stim_circuit_noisy: stim.Circuit,
    max_iter: int = 30,
    bp_method: str = "ps",
    schedule: str = "parallel",
    ms_scaling_factor: float = 1.0,
    osd_order: int = 2,
    osd_method: str = "osd_cs",
    error_channel: Optional[Sequence[float]] = None,
) -> Tuple[ldpc.BpOsdDecoder, np.ndarray]:
    """Build an ``ldpc.BpOsdDecoder`` from a noisy Stim circuit.

    BP-OSD = belief propagation + ordered statistics decoding. The
    BP step gets close to the optimal solution in the loopy graph;
    OSD post-processes the result and closes the gap. ``osd_order=2``
    is the standard practical choice for QEC and matches MWPM
    exactly at d=3 / p=0.001. Set ``osd_order=0`` to disable OSD and
    run plain BP (will be sub-optimal on surface codes).

    The decoder is reused across syndrome batches — build once per
    (distance, rounds, noise) and amortize. Returns the decoder plus
    the observable matrix ``O`` so callers can decode without
    rebuilding the factor graph.

    Parameters
    ----------
    stim_circuit_noisy : stim.Circuit WITH a noise model.
    max_iter           : int, BP iteration cap.
    bp_method          : 'ps' (product-sum), 'ms' (min-sum), or 'msl'.
    schedule           : 'parallel' or 'serial'.
    ms_scaling_factor  : scaling for min-sum variants.
    osd_order          : int, OSD search depth. 0 disables OSD, 2 is
                         a good default for QEC.
    osd_method         : OSD variant; 'osd_cs' (combinatorial sweep)
                         is the standard.
    error_channel      : optional sequence of per-error probabilities
                         (one entry per error mechanism in the DEM).
                         If provided, overrides the DEM-derived priors.
                         This is the calibration hook: feed it
                         Resonance-derived per-gate/per-qubit error
                         rates here. If None, the DEM defaults are used.
    """
    H, O, probs = dem_to_pcm(stim_circuit_noisy)
    if error_channel is None:
        channel = probs.tolist()
    else:
        channel = list(error_channel)
        if len(channel) != H.shape[1]:
            raise ValueError(
                f"error_channel has {len(channel)} entries, "
                f"expected {H.shape[1]} (one per DEM error mechanism)."
            )
    decoder = ldpc.BpOsdDecoder(
        H,
        error_channel=channel,
        max_iter=max_iter,
        bp_method=bp_method,
        schedule=schedule,
        ms_scaling_factor=ms_scaling_factor,
        osd_order=osd_order,
        osd_method=osd_method,
        input_vector_type="syndrome",
    )
    return decoder, O


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def decode_hardware_results_bp(
    syndromes: dict,
    stim_circuit_noisy: Optional[stim.Circuit] = None,
    decoder: Optional[ldpc.BpDecoder] = None,
    O: Optional[np.ndarray] = None,
    return_confidence: bool = True,
) -> Tuple[List[float], List[float], Optional[np.ndarray]]:
    """Decode syndromes with BP and return per-observable LER + 1-sigma.

    Parameters
    ----------
    syndromes          : dict from ``extract_syndromes()``.
    stim_circuit_noisy : noisy stim.Circuit, used to build the decoder
                         on the fly when ``decoder`` is None.
    decoder            : pre-built ldpc.BpDecoder. Preferred for sweeps.
    O                  : observable matrix from ``build_bp_decoder``.
                         Required if ``decoder`` is provided.
    return_confidence  : if True, also return per-shot, per-observable
                         soft confidence in [0, 1] (max over the two
                         observable outcomes, from the LLR of the
                         predicted error pattern).

    Returns
    -------
    ler       : list[float]  per-observable logical error rate.
    err       : list[float]  1-sigma binomial standard error per observable.
    confidence : np.ndarray  (shots, num_observables) float in [0, 1],
                 or None if ``return_confidence`` is False.
    """
    if decoder is None:
        if stim_circuit_noisy is None:
            raise ValueError(
                "decode_hardware_results_bp requires either `decoder` (and "
                "`O`) or `stim_circuit_noisy`."
            )
        decoder, O = build_bp_decoder(stim_circuit_noisy)

    det_events = syndromes["det_events"].astype(np.uint8)
    obs_flips = syndromes["obs_flips"].astype(bool)
    shots = len(obs_flips)
    num_obs = obs_flips.shape[1]

    # Decode each shot. ldpc has no built-in batch API at the time of
    # writing, so we loop. The inner call is C++ and fast (~us per shot
    # for d=3).
    predicted = np.zeros((shots, num_obs), dtype=np.uint8)
    confidence = np.zeros((shots, num_obs), dtype=np.float64) if return_confidence else None
    for i in range(shots):
        e_hat = decoder.decode(det_events[i])
        predicted[i] = (O @ e_hat) % 2
        if return_confidence:
            # Soft output: the LLR of the observable bit, derived from
            # the LLRs of the error pattern. LLR_obs = O @ LLR_e (with
            # the convention that positive LLR = "more likely 1").
            llr_e = decoder.log_prob_ratios
            llr_obs = O.astype(np.float64) @ llr_e  # (num_obs,)
            # Confidence in the prediction is the magnitude of the
            # predicted class's probability, mapped to [0, 1].
            p_obs = _sigmoid(llr_obs)
            confidence[i] = np.where(predicted[i] == 1, p_obs, 1.0 - p_obs)

    # LER + 1-sigma per observable.
    ler: List[float] = []
    err: List[float] = []
    for j in range(num_obs):
        wrong = int((predicted[:, j] != obs_flips[:, j]).sum())
        p = wrong / max(shots, 1)
        ler.append(float(p))
        err.append(float(np.sqrt(p * (1.0 - p) / max(shots, 1))))

    return ler, err, confidence
