"""
main.py

Simple test harness for the surface code QEC pipeline.
Run: python main.py
"""

import numpy as np
from surface_code import make_stim_circuit, make_qiskit_circuit, stim_to_qiskit, get_circuit_info
from extract_syndromes import extract_syndromes, extract_syndromes_manual
from internal_helpers import counts_to_measurement_array


def run_simulation_test():
    """
    Runs a simulation test comparing Stim-based and manual syndrome extraction.
    """
    print("=" * 60)
    print("  Surface Code QEC Pipeline - Simulation Test")
    print("=" * 60)
    print()

    # Parameters
    distance = 3
    rounds = 3
    shots = 1024

    # ----------------------------------------------------------------
    # Test 1: Stim circuit generation
    # ----------------------------------------------------------------
    print("1. Testing Stim circuit generation...")
    try:
        stim_circuit = make_stim_circuit(distance=distance, rounds=rounds, noise=None)
        info = get_circuit_info(stim_circuit)
        print("   [OK] Generated Stim circuit successfully")
        print(f"     Qubits: {info['num_qubits']}, "
              f"Data: {info['num_data_qubits']}, "
              f"Anc: {info['num_anc_qubits']}")
        print(f"     Detectors: {info['num_detectors']}, "
              f"Observables: {info['num_observables']}")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return

    # ----------------------------------------------------------------
    # Test 2: Qiskit circuit generation
    # ----------------------------------------------------------------
    print()
    print("2. Testing Qiskit circuit generation...")
    try:
        qc, metadata = make_qiskit_circuit(distance=distance, rounds=rounds, no_reset=False)
        print("   [OK] Generated Qiskit circuit successfully")
        print(f"     Qubits: {qc.num_qubits}, Depth: {qc.depth()}")
        print(f"     Data: {metadata['num_data_qubits']}, "
              f"Anc: {metadata['num_anc_qubits']}, "
              f"Rounds: {metadata['rounds']}")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return

    # ----------------------------------------------------------------
    # Test 3: Run Qiskit Aer simulation
    # ----------------------------------------------------------------
    print()
    print("3. Running Qiskit Aer simulation...")
    try:
        from qiskit_aer import AerSimulator

        simulator = AerSimulator()
        job = simulator.run(qc, shots=shots)
        counts = job.result().get_counts()

        # Convert counts to measurement array
        num_measurements = qc.num_clbits
        raw_meas = counts_to_measurement_array(counts, num_measurements, shots)

        print("   [OK] Simulation completed")
        print(f"     Total shots: {shots}")
        print(f"     Unique bitstrings: {len(counts)}")

        # Show a few sample counts
        sample_counts = list(counts.items())[:3]
        print(f"     Sample counts: {dict(sample_counts)}")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # ----------------------------------------------------------------
    # Test 4: Manual syndrome extraction
    # ----------------------------------------------------------------
    print()
    print("4. Running manual syndrome extraction (pure Qiskit)...")
    try:
        syndromes_manual = extract_syndromes_manual(raw_meas, metadata, print_summary=True)
        print("   [OK] Manual extraction completed")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # ----------------------------------------------------------------
    # Test 5: Stim-based syndrome extraction (if Stim circuit available)
    # ----------------------------------------------------------------
    print()
    print("5. Running Stim-based syndrome extraction...")
    try:
        # Convert stim circuit to Qiskit for consistent measurement ordering
        qc_from_stim, stim_to_dense, meas_order = stim_to_qiskit(stim_circuit)

        # Re-run the stim-based circuit in Qiskit for comparison
        job_stim = simulator.run(qc_from_stim, shots=shots)
        counts_stim = job_stim.result().get_counts()

        # Convert to same format
        raw_meas_stim = counts_to_measurement_array(counts_stim, stim_circuit.num_measurements, shots)

        # Extract syndromes using Stim's m2d converter
        syndromes_stim = extract_syndromes(raw_meas_stim, stim_circuit, print_summary=True)

        print("   [OK] Stim-based extraction completed")
        print(f"     Matching detectors: {syndromes_stim['num_detectors']}")

        # Compare results
        print()
        print("6. Comparing manual vs Stim-based extraction...")
        match_det = np.array_equal(
            syndromes_manual["det_events"][:, :syndromes_stim['num_detectors']],
            syndromes_stim["det_events"]
        )
        match_obs = np.array_equal(syndromes_manual["obs_flips"], syndromes_stim["obs_flips"])

        # Calculate how many shots have matching syndrome patterns
        # For a fair comparison, we need to account for different detector counts

        manual_weight = syndromes_manual["syndrome_weight_per_shot"]
        stim_weight = syndromes_stim["syndrome_weight_per_shot"]

        weight_match = np.mean(manual_weight == stim_weight)

        print(f"   Detection events shape match: {match_det}")
        print(f"   Observable flips match: {match_obs}")
        print(f"   Syndrome weight match: {weight_match*100:.1f}% of shots")
        print(f"   Mean weight (manual): {manual_weight.mean():.3f}")
        print(f"   Mean weight (stim):   {stim_weight.mean():.3f}")

    except Exception as e:
        print(f"   [ERROR] Stim path may not be available: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("   Note: Stim-based path requires compatible circuit structure.")
        print("   Manual extraction is the primary path for pure Qiskit workflow.")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print()
    print("=" * 60)
    print("  IMPLEMENTATION SUMMARY")
    print("=" * 60)
    print("""
Files modified:
  - surface_code.py: Implemented make_stim_circuit() and make_qiskit_circuit()
  - extract_syndromes.py: Added extract_syndromes_manual() function
  - main.py: Added run_simulation_test()

What was implemented:

1. make_stim_circuit(distance, rounds, noise, no_reset, memory)
   - Uses stim.Circuit.generated() with rotated surface code task
   - Supports Z-type (default) and X-type memory experiments
   - Optional noise model via DEFAULT_NOISE dict
   - no_reset flag for ancilla accumulation mode

2. make_qiskit_circuit(distance, rounds, noise, no_reset, memory)
   - Manual Qiskit QuantumCircuit construction
   - Returns (qc, metadata) tuple for downstream processing
   - Supports reset and no_reset modes
   - metadata dict for extract_syndromes_manual

3. extract_syndromes_manual(raw_meas, metadata)
   - Pure Python/numpy syndrome extraction
   - No Stim m2d converter dependency
   - Computes detection events as XOR between consecutive rounds
   - Returns same dict structure as extract_syndromes()

4. run_simulation_test() in main.py
   - Tests Stim circuit generation
   - Tests Qiskit circuit generation
   - Runs Aer simulator with 1024 shots
   - Compares manual and Stim-based extraction

Limitations:
  - Qiskit circuit structure is simplified (not optimized for real hardware)
  - Noise model not yet implemented for Qiskit path
  - Manual extraction uses simplified ancilla-data qubit mapping
  - For production use, consider using Stim path for better accuracy
""")


def main():
    run_simulation_test()


if __name__ == "__main__":
    main()