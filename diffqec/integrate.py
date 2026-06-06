"""Pipeline glue: satisfies the existing decode_hardware_results contract."""
from pathlib import Path
from typing import Optional
import numpy as np

from .decoder import load_decoder


def decode_hardware_results_diffqec(
    syndromes: dict,
    model_path: Optional[str] = None,
    syndrome_shape: Optional[tuple] = None,
    L: Optional[int] = None,
    **model_kwargs,
) -> tuple:
    """Decode hardware syndromes using a trained DiffQEC model.

    Parameters
    ----------
    syndromes    : dict from extract_syndromes()
    model_path   : path to checkpoint (final.pt)
    syndrome_shape : (rounds, D) expected by model
    L            : number of logical observables
    **model_kwargs : passed to DiffQEC constructor (e.g. hidden=16)

    Returns
    -------
    ler : list[float]  — per-observable logical error rate
    err : list[float]  — 1-sigma binomial standard error per observable
    """
    if model_path is None:
        raise ValueError("model_path must be provided for diffqec decoder")

    num_obs = syndromes["obs_flips"].shape[1]
    if L is None:
        L = num_obs
    if syndrome_shape is None:
        total_d = syndromes["num_detectors"]
        rounds_guess = int(round(total_d ** 0.5)) or 1
        D_guess = total_d // rounds_guess
        syndrome_shape = (rounds_guess, D_guess)

    decoder = load_decoder(model_path, syndrome_shape=syndrome_shape, L=L, **model_kwargs)
    predicted, _ = decoder.decode_syndromes(syndromes)
    actual = syndromes["obs_flips"].astype(bool)
    shots = len(actual)

    ler = []
    err = []
    for i in range(num_obs):
        errors = (predicted[:, i] != actual[:, i]).sum()
        p = float(errors) / max(shots, 1)
        ler.append(p)
        err.append((p * (1.0 - p) / max(shots, 1)) ** 0.5)

    return ler, err
