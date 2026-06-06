"""Data generation, parity split, and curriculum sampling."""
from typing import Tuple, Optional
import numpy as np
import stim
import torch
from torch.utils.data import Dataset, Sampler

from .noise import (
    NoiseModel,
    IQM_EMERALD_TYPICAL,
    apply_noise_to_circuit,
)


def generate_dem_samples(
    circuit: stim.Circuit,
    shots: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate detection events and observable flips via DEM sampling.

    Returns
    -------
    det_events : (shots, num_detectors) bool
    obs_flips  : (shots, num_observables) bool
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    det_events, obs_flips = sampler.sample(shots, separate_observables=True)
    return det_events, obs_flips


def make_test_circuit(
    distance: int = 3,
    rounds: int = 3,
    noise: float = 0.01,
) -> stim.Circuit:
    """Create a Stim rotated surface-code memory-Z circuit.

    .. deprecated::
        The single-knob phenomenological model is no longer recommended for
        training the decoder.  Use :func:`make_noisy_circuit` with a
        :class:`~diffqec.noise.NoiseModel` instead.
    """
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
    )


def make_noisy_circuit(
    distance: int = 3,
    rounds: int = 3,
    noise: Optional[NoiseModel] = None,
    *,
    memory: str = "Z",
    no_reset: bool = False,
) -> stim.Circuit:
    """Create a Stim rotated surface-code circuit with a hardware-style noise model.

    Unlike :func:`make_test_circuit`, this generator uses
    :func:`~diffqec.noise.apply_noise_to_circuit` to inject *per-instruction*
    error channels: DEPOLARIZE1 after every 1q Clifford, DEPOLARIZE2 after
    every 2q Clifford, X_ERROR after every measurement and reset, and a
    Z_ERROR on data qubits at the end of every round.

    Parameters
    ----------
    distance, rounds, memory, no_reset
        Identical semantics to :func:`make_test_circuit`.
    noise : NoiseModel, optional
        The hardware-calibrated noise model.  Defaults to
        :data:`IQM_EMERALD_TYPICAL` (typical 2024 IQM Emerald calibration).
        Pass ``noise=None`` to obtain a noiseless circuit.
    """
    code_task = (
        "surface_code:rotated_memory_x"
        if memory.upper() == "X"
        else "surface_code:rotated_memory_z"
    )
    base = stim.Circuit.generated(
        code_task=code_task,
        distance=distance,
        rounds=rounds,
    )
    if no_reset:
        base = stim.Circuit(str(base).replace("MR", "M"))
    if noise is None:
        return base
    return apply_noise_to_circuit(base, noise)


def generate_noisy_dem_samples(
    distance: int = 3,
    rounds: int = 3,
    shots: int = 10_000,
    noise: Optional[NoiseModel] = None,
    *,
    seed: Optional[int] = None,
    memory: str = "Z",
    no_reset: bool = False,
) -> Tuple[stim.Circuit, np.ndarray, np.ndarray]:
    """Generate a (circuit, det_events, obs_flips) triple with realistic noise.

    Returns
    -------
    circuit   : the noisy stim.Circuit that was sampled from
    det_events: (shots, num_detectors) bool
    obs_flips : (shots, num_observables) bool
    """
    if noise is None:
        noise = IQM_EMERALD_TYPICAL
    circuit = make_noisy_circuit(
        distance=distance,
        rounds=rounds,
        noise=noise,
        memory=memory,
        no_reset=no_reset,
    )
    sampler = circuit.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots, separate_observables=True)
    return circuit, det, obs


def noise_sweep_table(
    distance: int = 3,
    rounds: int = 3,
    shots: int = 5_000,
    base_noise: Optional[NoiseModel] = None,
    scales: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    *,
    seed: int = 0,
) -> list[dict]:
    """Run the DEM sampler at several ``scale * base_noise`` strengths.

    Returns one row per scale with the circuit, syndrome, observables, and
    detector firing rate summary.  This is the function the
    ``scripts/train_noisy_diffqec.py`` driver uses to materialise the
    LER-vs-noise curve on LUMI.
    """
    if base_noise is None:
        base_noise = IQM_EMERALD_TYPICAL
    rows: list[dict] = []
    for s in scales:
        noise_s = base_noise.scale(s)
        circ, det, obs = generate_noisy_dem_samples(
            distance=distance, rounds=rounds, shots=shots,
            noise=noise_s, seed=seed + int(s * 1000),
        )
        rows.append({
            "scale": s,
            "noise": noise_s,
            "circuit": circ,
            "det_events": det,
            "obs_flips": obs,
            "mean_detector_firing_rate": float(det.mean()),
        })
    return rows


def reshape_syndrome(
    det_events: np.ndarray,
    circuit: stim.Circuit,
    target_rounds: Optional[int] = None,
) -> np.ndarray:
    """Reshape flat detection events into (shots, rounds, D) for the model.

    Stim detectors are ordered round-by-round. We attempt to infer the
    number of rounds from the circuit and split accordingly. If the
    split is not exact, we pad/truncate to target_rounds * max_det_per_round.
    """
    shots, total_d = det_events.shape
    # Heuristic: for a rotated surface code with R rounds,
    # detectors ≈ (R-1)*num_ancilla + num_data.
    # We simply divide by target_rounds if provided, otherwise use sqrt heuristic.
    if target_rounds is None:
        target_rounds = int(round(total_d ** 0.5)) or 1
    d_per_round = total_d // target_rounds
    if d_per_round * target_rounds < total_d:
        # pad to multiple
        pad = d_per_round * target_rounds + target_rounds - total_d
        det_events = np.pad(det_events, ((0, 0), (0, pad)), mode="constant")
        d_per_round = (total_d + pad) // target_rounds
    # truncate to exact multiple
    usable = target_rounds * d_per_round
    det_events = det_events[:, :usable]
    return det_events.reshape(shots, target_rounds, d_per_round)


class ParityDataset(Dataset):
    """Train/eval split by even/odd shot index.

    Parameters
    ----------
    det_events    : (shots, num_detectors) bool or float
    obs_flips     : (shots, num_observables) bool or float
    parity        : "even" or "odd"
    target_rounds : optional int. If provided, flattens each shot to
                    (target_rounds, D) where D = num_detectors // target_rounds.
    """

    def __init__(
        self,
        det_events: np.ndarray,
        obs_flips: np.ndarray,
        parity: str = "even",
        target_rounds: int | None = None,
    ):
        assert parity in ("even", "odd")
        mask = np.arange(len(det_events)) % 2 == (0 if parity == "even" else 1)
        det = det_events[mask].astype(np.float32)
        if target_rounds is not None:
            d_per_round = det.shape[1] // target_rounds
            usable = target_rounds * d_per_round
            det = det[:, :usable].reshape(len(det), target_rounds, d_per_round)
        self.det_events = det
        self.obs_flips = obs_flips[mask].astype(np.float32)
        self.num_detectors = self.det_events.shape[1]

    def __len__(self) -> int:
        return len(self.det_events)

    def __getitem__(self, idx: int):
        return {
            "syndrome": torch.from_numpy(self.det_events[idx]),
            "logical": torch.from_numpy(self.obs_flips[idx]),
        }


class CurriculumSampler(Sampler):
    """Sample batches with round counts drawn from a curriculum schedule.

    For simplicity we assume the dataset already contains the *maximum*
    number of rounds and we slice the prefix on the fly. If the dataset
    has varying round counts, this sampler groups by round count.
    """

    def __init__(
        self,
        dataset: Dataset,
        round_schedule: Tuple[int, ...] = (3, 5, 7, 9),
        epoch_to_max_rounds: Optional[dict] = None,
    ):
        self.dataset = dataset
        self.round_schedule = sorted(round_schedule)
        self.epoch_to_max_rounds = epoch_to_max_rounds or {}
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    @property
    def max_rounds(self) -> int:
        default = self.round_schedule[-1]
        allowed = [r for r in self.round_schedule
                   if self.epoch_to_max_rounds.get(self.epoch, default) >= r]
        return max(allowed) if allowed else self.round_schedule[0]

    def __iter__(self):
        # For simplicity just yield all indices; the collate function in
        # training.py will slice syndromes to max_rounds. In a production
        # setting you would group indices by compatible round counts.
        yield from range(len(self.dataset))

    def __len__(self) -> int:
        return len(self.dataset)
