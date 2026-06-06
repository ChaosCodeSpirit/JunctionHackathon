"""Qiskit-Aer stabilizer simulation of the 54-qubit surface code.

We use the `stabilizer` method (Clifford-only) which scales easily to 54
qubits and beyond, with polynomial memory in circuit depth.

For QEC, the stabilizer simulator is the natural choice because surface
codes are Clifford circuits.  We inject a simple symmetric depolarizing
noise channel after each CX and each measurement.

Output format: a dict matching `extract_syndromes.extract_syndromes()` so
the existing DiffQEC and PyMatching decoders plug in without changes.
"""
from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    pauli_error,
)

from .circuit import (
    SurfaceCode54Q,
    build_memory_z_circuit,
    syndrome_indices,
    N_QUBITS,
)


def build_depolarizing_noise_model(p: float) -> NoiseModel:
    """Symmetric depolarizing noise matching the IQM Q50 baseline.

    p_1q : 1-qubit gate error (default: p)
    p_2q : 2-qubit gate error (default: 10 * p) — IQM CZ ~10x worse than PRX
    p_meas: measurement flip probability (default: p)
    p_reset: reset error (default: p)
    """
    p_1q = p
    p_2q = min(10 * p, 0.5)
    p_meas = p
    p_reset = p
    noise = NoiseModel()
    # 1-qubit depolarizing on H, X, Z — apply to all qubits
    err_1q = depolarizing_error(p_1q, 1)
    for g in ("h", "x", "z", "sx"):
        noise.add_all_qubit_quantum_error(err_1q, g)
    # 2-qubit depolarizing on CX / CZ (all-to-all)
    err_2q = depolarizing_error(p_2q, 2)
    noise.add_all_qubit_quantum_error(err_2q, ["cx", "cz"])
    # Measurement + reset
    noise.add_all_qubit_quantum_error(
        pauli_error([("X", p_meas), ("I", 1 - p_meas)]), "measure"
    )
    noise.add_all_qubit_quantum_error(
        pauli_error([("X", p_reset), ("I", 1 - p_reset)]), "reset"
    )
    return noise


def run_stabilizer_simulation(
    spec: SurfaceCode54Q,
    shots: int = 4096,
    seed: int = 42,
) -> dict:
    """Run the 54-qubit surface code on Qiskit-Aer stabilizer simulator.

    Returns a dict shaped like `extract_syndromes.extract_syndromes()` so it
    can be consumed directly by `diffqec.integrate.decode_hardware_results_diffqec`
    or by the existing PyMatching decoder.
    """
    qc = build_memory_z_circuit(spec)
    noise = build_depolarizing_noise_model(spec.p)

    # Stabilizer method handles Clifford + noise channels naturally
    sim = AerSimulator(method="stabilizer", noise_model=noise, seed_simulator=seed)
    qc_t = transpile(qc, sim, optimization_level=1)

    job = sim.run(qc_t, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Convert counts → (shots, n_qubits) bool matrix
    raw_meas = _counts_to_array(counts, spec.n_qubits, shots)

    # Re-derive detection events and observable flips.
    # In a single-round memory experiment with reset-then-measure ancilla:
    #   - The X-ancilla measurement at round r is the syndrome bit for that stabilizer.
    #   - The data measurement parity (XOR of all data bits) is the logical observable.
    det_events, obs_flips, syndrome_weight, firing_rates = _extract_syndromes(
        raw_meas, spec
    )

    return {
        "det_events": det_events,
        "obs_flips": obs_flips,
        "raw_meas": raw_meas,
        "num_detectors": det_events.shape[1],
        "num_shots": shots,
        "syndrome_weight_per_shot": syndrome_weight,
        "detector_firing_rate": firing_rates,
        "spec": spec,
        "noise_model": noise,
    }


def _counts_to_array(counts: dict, n_bits: int, shots: int) -> np.ndarray:
    """Convert Qiskit counts dict to a (shots, n_bits) bool array.

    Qiskit bitstrings are MSB-first; we reverse to LSB-first (qubit 0 = rightmost).
    """
    arr = np.zeros((shots, n_bits), dtype=bool)
    i = 0
    for bitstring, count in counts.items():
        # Pad to n_bits
        s = bitstring.zfill(n_bits)
        bits = np.array([int(b) for b in s[::-1]], dtype=bool)  # LSB first
        arr[i : i + count] = bits
        i += count
    return arr


def _extract_syndromes(raw_meas: np.ndarray, spec: SurfaceCode54Q):
    """Derive detection events and observable flips from raw measurements.

    For a single-round memory-Z experiment:
      - Detection events: which X-ancilla bits fired (i.e., measured 1).
        (In a real Stim-style experiment we'd XOR with the previous round's
        ancilla measurement.  For single-round, the ancilla bit itself IS
        the detection event.)
      - Observable flip: parity of all data measurements (memory-Z logical).
    """
    idx = syndrome_indices(spec)
    x_anc = idx["x_ancilla"]
    data = idx["data"]
    flags = idx["flags"]

    det_events = raw_meas[:, x_anc]  # (shots, 24)
    syndrome_weight = det_events.sum(axis=1).astype(int)
    firing_rates = det_events.mean(axis=0)

    # Observable: XOR of all data measurement bits
    data_parity = raw_meas[:, data].sum(axis=1) % 2
    obs_flips = data_parity[:, None].astype(bool)  # (shots, 1)

    return det_events, obs_flips, syndrome_weight, firing_rates


if __name__ == "__main__":
    spec = SurfaceCode54Q(distance=5, rounds=1, p=0.001)
    out = run_stabilizer_simulation(spec, shots=128, seed=0)
    print(f"Det events shape: {out['det_events'].shape}")
    print(f"Obs flips shape:  {out['obs_flips'].shape}")
    print(f"Mean syndrome wt: {out['syndrome_weight_per_shot'].mean():.2f}")
    print(f"Logical flip rate:{out['obs_flips'].mean():.4f}")
