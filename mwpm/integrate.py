"""MWPM decoder — minimum-weight perfect matching via PyMatching.

Thin pipeline glue. The decoder is just two operations:

    1. build_matching_decoder(circuit)         # graph construction
    2. matcher.decode_batch(det_events)        # batched matching

Anything beyond that (data helpers, model classes, confidence scores)
does not exist for MWPM. DiffQEC needs a class because the model is
stateful; PyMatching's ``Matching`` is already the state.
"""
from typing import List, Optional, Tuple
import numpy as np
import pymatching
import stim

__all__ = ["build_matching_decoder", "decode_hardware_results_mwpm"]


def build_matching_decoder(
    stim_circuit_noisy: stim.Circuit,
    decompose_errors: bool = True,
) -> pymatching.Matching:
    """Build a PyMatching MWPM decoder from a noisy Stim circuit's DEM.

    Edge weights are ``w = -ln(p / (1-p))`` from the DEM's per-error
    probabilities. Build once per (distance, rounds, noise) and reuse
    across batches — this is the expensive step.
    """
    dem = stim_circuit_noisy.detector_error_model(decompose_errors=decompose_errors)
    return pymatching.Matching.from_detector_error_model(dem)


def decode_hardware_results_mwpm(
    syndromes: dict,
    stim_circuit_noisy: Optional[stim.Circuit] = None,
    matcher: Optional[pymatching.Matching] = None,
) -> Tuple[List[float], List[float]]:
    """Decode syndromes with MWPM and return per-observable LER + 1-sigma.

    Parameters
    ----------
    syndromes          : dict from ``extract_syndromes()``.
    stim_circuit_noisy : noisy stim.Circuit, used to build the matcher
                         on the fly when ``matcher`` is None.
    matcher            : pre-built ``pymatching.Matching``. Preferred for
                         sweeps so the graph is amortized.

    Returns
    -------
    ler : list[float]  per-observable logical error rate.
    err : list[float]  1-sigma binomial standard error per observable.
    """
    if matcher is None:
        if stim_circuit_noisy is None:
            raise ValueError(
                "decode_hardware_results_mwpm requires either `matcher` or "
                "`stim_circuit_noisy`."
            )
        matcher = build_matching_decoder(stim_circuit_noisy)

    det_events = syndromes["det_events"].astype(np.uint8)
    obs_flips = syndromes["obs_flips"].astype(bool)

    predicted = matcher.decode_batch(det_events).astype(bool)
    shots = len(obs_flips)
    num_obs = obs_flips.shape[1]

    ler: List[float] = []
    err: List[float] = []
    for i in range(num_obs):
        wrong = int((predicted[:, i] != obs_flips[:, i]).sum())
        p = wrong / max(shots, 1)
        ler.append(float(p))
        err.append(float(np.sqrt(p * (1.0 - p) / max(shots, 1))))
    return ler, err
