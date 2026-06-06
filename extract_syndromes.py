
import numpy as np
import stim


def extract_syndromes_manual(
    raw_meas: np.ndarray,
    metadata: dict,
    print_summary: bool = True,
) -> dict:
    """
    Manual syndrome extraction for pure-Qiskit workflow without Stim's m2d converter.

    Detection events are computed as the XOR of consecutive measurements of the
    same ancilla qubit (a stabilizer changes between rounds indicates an error).
    The logical observable is recovered from the parity of the final data qubit
    measurements.

    Parameters
    ----------
    raw_meas : np.ndarray
        Shape (shots, num_measurements), dtype bool. Direct output of running
        the Qiskit circuit on a simulator.
    metadata : dict
        Dictionary from make_qiskit_circuit() containing:
        - num_data_qubits: number of data qubits
        - num_anc_qubits: number of ancilla qubits
        - data_qubit_indices: list of data qubit indices
        - anc_qubit_indices: list of ancilla qubit indices
        - rounds: number of measurement rounds
        - memory: "Z" or "X"
    print_summary : bool, optional
        Print a human-readable syndrome summary. Default is True.

    Returns
    -------
    dict with keys:
        "det_events"   : np.ndarray (shots, num_detectors), bool
                         Detection events — the syndrome the decoder consumes.
        "obs_flips"    : np.ndarray (shots, num_observables), bool
                         Logical observable outcomes from final data readout.
        "raw_meas"     : np.ndarray (shots, num_measurements), bool
                         The original raw measurements, passed through unchanged.
        "num_detectors": int
        "num_shots"    : int
        "syndrome_weight_per_shot" : np.ndarray (shots,), int
                         Number of detectors that fired per shot.
        "detector_firing_rate" : np.ndarray (num_detectors,), float
                         Per-detector fraction of shots in which it fired.
    """
    shots = len(raw_meas)
    num_meas = raw_meas.shape[1]

    num_data = metadata["num_data_qubits"]
    num_anc = metadata["num_anc_qubits"]
    rounds = metadata["rounds"]
    anc_indices = metadata["anc_qubit_indices"]
    data_indices = metadata["data_qubit_indices"]

    # The measurement order from Qiskit:
    # First rounds * num_anc measurements are ancilla measurements (one per round)
    # Then num_data measurements are final data qubit measurements

    # Build mapping: for each ancilla and each round, which measurement column
    # corresponds to that ancilla's measurement in that round
    num_detectors = num_anc * (rounds - 1)  # Detection events between consecutive rounds

    det_events = np.zeros((shots, num_detectors), dtype=bool)
    det_idx = 0

    for anc_idx, anc in enumerate(anc_indices):
        for round_idx in range(rounds - 1):
            # Measurement column for this ancilla at this round
            meas_round = raw_meas[:, anc_idx + round_idx * num_anc]
            meas_next = raw_meas[:, anc_idx + (round_idx + 1) * num_anc]
            # Detection event = XOR (measurement changed between rounds)
            det_events[:, det_idx] = np.logical_xor(meas_round, meas_next)
            det_idx += 1

    # Logical observable from parity of final data qubit measurements
    # Data measurements start at column rounds * num_anc
    data_meas_start = rounds * num_anc
    data_measurements = raw_meas[:, data_meas_start:data_meas_start + num_data]

    # Parity of data measurements = logical observable
    # For Z-memory: logical Z = parity of all data qubits (XOR of all)
    obs_flips = np.zeros((shots, 1), dtype=bool)
    obs_flips[:, 0] = np.sum(data_measurements, axis=1) % 2

    # Compute syndrome weights and firing rates
    weights = det_events.sum(axis=1).astype(int)
    firing_rates = det_events.mean(axis=0)

    if print_summary:
        print("-" * 56)
        print("  Manual Syndrome extraction summary")
        print("-" * 56)
        print(f"  Shots              : {shots}")
        print(f"  Num detectors      : {num_detectors}")
        print(f"  Num observables    : 1")
        print(f"  Mean syndrome wt   : {weights.mean():.3f}  "
              f"(max possible {num_detectors})")
        print(f"  Shots w/ 0 errors  : {int((weights == 0).sum())}  "
              f"({100*(weights==0).mean():.1f}%)")
        print(f"  Shots w/ >=1 error : {int((weights > 0).sum())}  "
              f"({100*(weights>0).mean():.1f}%)")
        print(f"  Logical flip rate  : {obs_flips.mean(axis=0)}")
        print()
        print("  Per-detector firing rates (flag anything > 0.2 as hot):")
        for i, r in enumerate(firing_rates):
            flag = "  <-- HOT" if r > 0.2 else ""
            print(f"    detector {i:>3}: {r:.4f}{flag}")
        print("-" * 56)

    return {
        "det_events"              : det_events,
        "obs_flips"               : obs_flips,
        "raw_meas"                : raw_meas,
        "num_detectors"           : num_detectors,
        "num_shots"               : shots,
        "syndrome_weight_per_shot": weights,
        "detector_firing_rate"    : firing_rates,
    }


def extract_syndromes(
    raw_meas: np.ndarray,
    stim_circuit_noisy: stim.Circuit,
    print_summary: bool = True,
) -> dict:
    """
    Converts raw hardware measurements into detection events and observable
    flips. Sits between run_hardware_experiment() and any decoder (PyMatching
    or NVIDIA Ising Predecoder).

    Parameters
    ----------
    raw_meas            : np.ndarray  shape (shots, num_measurements), bool
                          Direct output of run_hardware_experiment().
    stim_circuit_noisy  : stim.Circuit  WITH noise model.
                          Used to define the detector structure for m2d.
                          Use make_stim_circuit(d, rounds, DEFAULT_NOISE).
    print_summary       : bool  print a human-readable syndrome summary.

    Returns
    -------
    dict with keys:
        "det_events"   : np.ndarray (shots, num_detectors), bool
                         Detection events — the syndrome the decoder consumes.
                         True = stabilizer fired (changed relative to reference).
        "obs_flips"    : np.ndarray (shots, num_observables), bool
                         Logical observable outcomes from the final data readout.
                         True = logical flip observed.
        "raw_meas"     : np.ndarray (shots, num_measurements), bool
                         The original raw measurements, passed through unchanged.
        "num_detectors": int
        "num_shots"    : int
        "syndrome_weight_per_shot" : np.ndarray (shots,), int
                         Number of detectors that fired per shot.
                         Low mean = low error rate; mean near num_detectors/2
                         indicates the decoder is seeing noise at or above threshold.
        "detector_firing_rate" : np.ndarray (num_detectors,), float
                         Per-detector fraction of shots in which it fired.
                         Useful for spotting hot qubits or miscalibrated ancillas.
    """
    converter  = stim_circuit_noisy.compile_m2d_converter()
    det_events, obs_flips = converter.convert(
        measurements=raw_meas.astype(bool),
        separate_observables=True,
    )

    shots         = len(raw_meas)
    n_det         = det_events.shape[1]
    weights       = det_events.sum(axis=1).astype(int)   # fired detectors per shot
    firing_rates  = det_events.mean(axis=0)               # per-detector firing rate

    # if print_summary:
    #     print("─" * 56)
    #     print("  Syndrome extraction summary")
    #     print("─" * 56)
    #     print(f"  Shots              : {shots}")
    #     print(f"  Num detectors      : {n_det}")
    #     print(f"  Num observables    : {obs_flips.shape[1]}")
    #     print(f"  Mean syndrome wt   : {weights.mean():.3f}  "
    #           f"(max possible {n_det})")
    #     print(f"  Shots w/ 0 errors  : {int((weights == 0).sum())}  "
    #           f"({100*(weights==0).mean():.1f}%)")
    #     print(f"  Shots w/ ≥1 error  : {int((weights > 0).sum())}  "
    #           f"({100*(weights>0).mean():.1f}%)")
    #     print(f"  Logical flip rate  : "
    #           f"{obs_flips.mean(axis=0)} (per observable)")
    #     print()
    #     print("  Per-detector firing rates (flag anything > 0.2 as hot):")
    #     for i, r in enumerate(firing_rates):
    #         flag = "  ← HOT" if r > 0.2 else ""
    #         print(f"    detector {i:>3}: {r:.4f}{flag}")
    #     print("─" * 56)

    return {
        "det_events"              : det_events,
        "obs_flips"               : obs_flips,
        "raw_meas"                : raw_meas,
        "num_detectors"           : n_det,
        "num_shots"               : shots,
        "syndrome_weight_per_shot": weights,
        "detector_firing_rate"    : firing_rates,
    }
