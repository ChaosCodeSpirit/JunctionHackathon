# -*- coding: utf-8 -*-

from internal_helpers import *

"""
surface_code_stim.py

Full pipeline: (Stim circuit →) Qiskit → IQM Emerald → Decoder

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Code: rotated surface code, memory-Z experiment                    │
  │  Distance 3: 9 data + 8 ancilla = 17 qubits                         │
  │  Connectivity: diagonal,                                            │
  │  SWAPs for 6 of the 12 unique CX pairs. Circuit depth ≈ 9 (ideal).  │
  └─────────────────────────────────────────────────────────────────────┘

Sections
--------
  1. Stim circuit generation
  2. Stim → Qiskit conversion
  3. OPTIONAL Emerald qubit mapping
  4. Stim simulation  (no hardware, for threshold / LER curves)
  5. Hardware execution on IQM Resonance
  6. Hardware results → detection events → Decoder

Usage
-----
  # Simulation only:
  from surface_code_stim import simulate_ler

  # Full hardware pipeline:
  from surface_code_stim import run_hardware_experiment


"""

import numpy as np
import stim
import pymatching
from qiskit import QuantumCircuit, transpile
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  1.  STIM CIRCUIT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

#: Default noise model matching IQM Emerald typical gate error rates (~0.1-0.5%).
#: Adjust these based on your latest calibration data from Resonance.
DEFAULT_NOISE = dict(
    after_clifford_depolarization  = 0.003,
    after_reset_flip_probability   = 0.003,
    before_measure_flip_probability= 0.003,
    before_round_data_depolarization = 0.003,
)


def make_stim_circuit(
    distance: int = 3,
    rounds: int = 3,
    noise: dict | None = None,
    no_reset: bool = False,
    memory: str = "Z",
) -> stim.Circuit:
    """
    Generates a Stim circuit for the rotated surface code memory experiment.

    Uses Stim's built-in surface code generator for correctness. Supports both
    Z-type (default) and X-type memory experiments.

    Parameters
    ----------
    distance : int, optional
        Code distance. Default is 3.
    rounds : int, optional
        Number of stabilizer measurement rounds. Default is 3.
    noise : dict | None, optional
        If provided, applies noise model with keys:
        - after_clifford_depolarization
        - after_reset_flip_probability
        - before_measure_flip_probability
        - before_round_data_depolarization
        If None, returns a noiseless circuit.
    no_reset : bool, optional
        If True, uses M (measure only) instead of MR (measure + reset) for ancillas.
        Default is False.
    memory : str, optional
        Memory type: "Z" for Z-type (default) or "X" for X-type.

    Returns
    -------
    stim.Circuit
        A Stim circuit with DETECTOR and OBSERVABLE_INCLUDE annotations.
    """
    # Choose the correct code task for Stim's generator
    if memory.upper() == "X":
        code_task = "surface_code:rotated_memory_x"
    else:
        code_task = "surface_code:rotated_memory_z"

    # Generate the base circuit using Stim's built-in generator
    circuit = stim.Circuit.generated(
        code_task=code_task,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise.get("after_clifford_depolarization", 0) if noise else 0,
        after_reset_flip_probability=noise.get("after_reset_flip_probability", 0) if noise else 0,
        before_measure_flip_probability=noise.get("before_measure_flip_probability", 0) if noise else 0,
        before_round_data_depolarization=noise.get("before_round_data_depolarization", 0) if noise else 0,
    )

    # If no_reset is requested, we need to transform MR -> M
    # This requires rebuilding the circuit without resets on ancillas
    if no_reset:
        # Build a new circuit string with MR replaced by M
        circuit_str = str(circuit)
        # Replace MR with M in the string representation
        circuit_str = circuit_str.replace("MR", "M")
        circuit = stim.Circuit(circuit_str)

    return circuit 

def get_circuit_info(circuit: stim.Circuit) -> dict:
    """
    Returns a summary dict of qubit counts and measurement structure.
    Useful for sanity-checking before hardware runs.
    """
    data_q, anc_q = get_qubit_lists(circuit)
    return {
        "num_qubits"      : circuit.num_qubits,
        "num_data_qubits" : len(data_q),
        "num_anc_qubits"  : len(anc_q),
        "num_measurements": circuit.num_measurements,
        "num_detectors"   : circuit.num_detectors,
        "num_observables" : circuit.num_observables,
        "data_stim_indices": sorted(data_q),
        "anc_stim_indices" : sorted(anc_q),
        "meas_order"      : get_meas_order(circuit),
    }


def make_qiskit_circuit(
    distance: int = 3,
    rounds: int = 3,
    noise: dict | None = None,
    no_reset: bool = False,
    memory: str = "Z",
) -> tuple[QuantumCircuit, dict]:
    """
    Generates a Qiskit circuit for the rotated surface code memory experiment.

    Builds the circuit manually without Stim dependencies. For distance 3, uses
    9 data qubits and 8 ancilla qubits (17 total) arranged in a rotated lattice.

    Parameters
    ----------
    distance : int, optional
        Code distance. Default is 3.
    rounds : int, optional
        Number of stabilizer measurement rounds. Default is 3.
    noise : dict | None, optional
        Not yet implemented for Qiskit (reserved for future noise model pass).
        Default is None (noiseless).
    no_reset : bool, optional
        If True, ancillas accumulate measurements across rounds (no mid-circuit reset).
        Default is False.
    memory : str, optional
        Memory type: "Z" for Z-type stabilizers (default) or "X" for X-type.

    Returns
    -------
    qc : QuantumCircuit
        Qiskit QuantumCircuit with the surface code experiment.
    metadata : dict
        Dictionary containing:
        - num_data_qubits: number of data qubits
        - num_anc_qubits: number of ancilla qubits
        - data_qubit_indices: list of data qubit indices
        - anc_qubit_indices: list of ancilla qubit indices
        - rounds: number of measurement rounds
        - memory: memory type ("Z" or "X")
    """
    # For rotated surface code: distance d has d^2 data qubits and (d^2 - 1) ancillas
    d = distance
    num_data = d * d
    num_anc = num_data - 1  # d^2 - 1 ancillas for rotated code

    # Total qubits = data + ancilla
    total_qubits = num_data + num_anc

    # Total classical bits needed:
    # - rounds * num_anc measurements for ancillas (one per round)
    # - num_data measurements for final data qubit readout
    total_clbits = rounds * num_anc + num_data

    # Create the quantum circuit
    # We use classical bits for all measurements (stored separately)
    qc = QuantumCircuit(total_qubits, total_clbits)

    # Define data qubit indices (0 to num_data-1)
    data_indices = list(range(num_data))

    # Define ancilla qubit indices (num_data to num_data + num_anc - 1)
    anc_indices = list(range(num_data, total_qubits))

    # Helper function to get grid position from linear index
    # For rotated surface code, data qubits at (i, j) where i,j are 0,1,2,...,d-1
    # and they are offset so that actual coordinates are 2*i+1, 2*j+1 (odd positions)
    def data_qubit_coords(idx):
        """Returns (x, y) grid coordinates for data qubit idx"""
        i = idx // d
        j = idx % d
        return (2 * i + 1, 2 * j + 1)  # odd coordinates for data qubits

    def anc_qubit_coords(idx):
        """Returns (x, y) grid coordinates for ancilla qubit idx"""
        # Ancillas are at plaquette centers: even coordinates
        # There are (d-1) x (d-1) plaquettes for Z-type and (d+1) x (d+1) for X-type
        # For Z-type: ancilla i,j corresponds to plaquette between data qubits
        base_idx = idx - num_data
        # For simplicity, assign ancillas to plaquette positions
        # Z-type: (d-1) x (d-1) plaquettes with Z stabilizers
        # Layout: ancilla at (2*i, 2*j) for i,j in 0..d-2
        i = base_idx // (d - 1) if d > 1 else 0
        j = base_idx % (d - 1) if d > 1 else 0
        return (2 * i, 2 * j)  # even coordinates for ancillas

    # Build the circuit round by round
    for round_idx in range(rounds):
        # Reset ancillas at the start of each round (unless no_reset)
        if not no_reset:
            for anc in anc_indices:
                qc.reset(anc)

        # Apply stabilizer measurements for this round
        # For Z-type memory: measure Z stabilizers ( plaquettes)
        # Each Z stabilizer measures 4 neighboring data qubits via CX/CZ

        if memory.upper() == "Z":
            # Z-type stabilizers: measure parity of 4 data qubits around each ancilla
            # For each ancilla, apply CX from data qubits to ancilla
            for anc_idx, anc in enumerate(anc_indices):
                # Get the 4 data qubits that belong to this plaquette
                # For a simple rotated layout, we pair ancillas with nearby data
                # This is a simplified pairing - proper implementation would use
                # the actual rotated code geometry

                # For a more structured approach, use the plaquette structure:
                # Each ancilla at (2*i, 2*j) measures data qubits at:
                # (2*i+1, 2*j-1), (2*i+1, 2*j+1), (2*i-1, 2*j+1), (2*i-1, 2*j-1)
                # But for d=3 with proper indexing, we simplify:

                # Simple mapping: ancilla k measures data qubits (k, k+1, k+2, k+3) mod num_data
                # This is a simplification for demonstration
                base = (anc_idx * 4) % num_data
                for offset in range(4):
                    data_idx = (base + offset) % num_data
                    qc.cx(data_indices[data_idx], anc)
        else:
            # X-type stabilizers: apply H before and after, measure in X basis
            # For X-type: apply H to ancillas, then CZ to data, then H again
            for anc in anc_indices:
                qc.h(anc)
            for anc_idx, anc in enumerate(anc_indices):
                base = (anc_idx * 4) % num_data
                for offset in range(4):
                    data_idx = (base + offset) % num_data
                    qc.cz(data_indices[data_idx], anc)
            for anc in anc_indices:
                qc.h(anc)

        # Measure ancillas into distinct classical bits per round
        for i, anc in enumerate(anc_indices):
            clbit = round_idx * num_anc + i
            qc.measure(anc, clbit)

        # Add barrier for clarity between rounds
        if round_idx < rounds - 1:
            qc.barrier()

    # Final data qubit measurement (logical observable)
    # Data measurements start after all ancilla measurements
    data_meas_start = rounds * num_anc
    for data_idx, data in enumerate(data_indices):
        qc.measure(data, data_meas_start + data_idx)

    # Build metadata
    metadata = {
        "num_data_qubits": num_data,
        "num_anc_qubits": num_anc,
        "data_qubit_indices": data_indices,
        "anc_qubit_indices": anc_indices,
        "rounds": rounds,
        "memory": memory.upper(),
        "no_reset": no_reset,
    }

    return qc, metadata 

# ─────────────────────────────────────────────────────────────────────────────
#  2.  STIM → QISKIT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

# Instructions that exist only in Stim's simulation model — skip them when
# building the Qiskit circuit for hardware.
_STIM_SKIP = frozenset({
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "Y_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR",
    "ELSE_CORRELATED_ERROR", "DETECTOR", "OBSERVABLE_INCLUDE",
    "QUBIT_COORDS", "SHIFT_COORDS", "TICK",
})


def stim_to_qiskit(stim_circuit: stim.Circuit) -> tuple[QuantumCircuit, dict, list]:
    """
    Converts a (noiseless) Stim circuit into a Qiskit QuantumCircuit.

    The Stim circuit is first flattened to unroll any REPEAT blocks,
    then noise/annotation instructions are dropped. The resulting
    Qiskit circuit is suitable for transpilation onto IQM Resonance.

    Parameters
    ----------
    stim_circuit : stim.Circuit  (use make_stim_circuit(..., noise=None))

    Returns
    -------
    qc            : QuantumCircuit  with dense qubit indices 0..N-1
    stim_to_dense : dict  stim_qubit_index → dense_qiskit_index
    meas_order    : list[int]  stim qubit indices in measurement order
                    Use this to map hardware bitstrings back to Stim's
                    measurement record for detection event conversion.
    """
    flat = stim_circuit.flattened()

    # Build dense re-indexing: Stim uses sparse indices (e.g. 1,3,5,8,…,25)
    # Qiskit needs contiguous 0..N-1 indices.
    data_q, anc_q = get_qubit_lists(stim_circuit)
    all_stim_q    = sorted(set(data_q + anc_q))
    stim_to_dense = {sq: i for i, sq in enumerate(all_stim_q)}

    n_qubits = len(all_stim_q)
    n_clbits = stim_circuit.num_measurements # one classical bit per measured qubit
    qc = QuantumCircuit(n_qubits, n_clbits)

    meas_idx  = 0
    meas_order = []

    for instr in flat:
        name = instr.name

        if name in _STIM_SKIP:
            continue

        # Extract qubit targets (ignore measurement-record targets in M gates)
        targets = [t.value for t in instr.targets_copy() if t.is_qubit_target]
        dense   = [stim_to_dense[t] for t in targets]

        if name == "R":
            for d in dense:
                qc.reset(d)

        elif name == "H":
            for d in dense:
                qc.h(d)

        elif name == "CX":
            for i in range(0, len(dense), 2):
                qc.cx(dense[i], dense[i + 1])

        elif name == "CZ":
            for i in range(0, len(dense), 2):
                qc.cz(dense[i], dense[i + 1])

        elif name == "X":
            for d in dense:
                qc.x(d)

        elif name in ("M", "MR"):
            for stim_q, d in zip(targets, dense):
                qc.measure(d, meas_idx)
                meas_order.append(stim_q)
                meas_idx += 1
            # MR = measure-then-reset (mid-circuit reset for multi-round)
            if name == "MR":
                for d in dense:
                    qc.reset(d)

        else:
            # Warn about genuinely unhandled gates (not noise/annotations)
            raise NotImplementedError(
                f"[stim_to_qiskit] Unhandled gate '{name}'. "
                f"If this is a noise/annotation instruction, add it to _STIM_SKIP. "
                f"If it is a gate instruction, implement it explicitly."
                )

    return qc, stim_to_dense, meas_order


# ─────────────────────────────────────────────────────────────────────────────
#  3.  EMERALD QUBIT MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  4.  STIM/AER SIMULATION  (no hardware needed)
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  5.  Hardware execution
# ─────────────────────────────────────────────────────────────────────────────

# TBD

# ─────────────────────────────────────────────────────────────────────────────
#  6.  Offline Decoding 
# ─────────────────────────────────────────────────────────────────────────────

# TBD