from surface_code import * 
from build_emerald_qubit_rotated import *
from extract_syndromes import *


def get_token():
    with open("TOKEN.txt") as f:
        token = f.readlines()[0]
        return token


# ─────────────────────────────────────────────────────────────────────────────
#  5.  HARDWARE EXECUTION ON IQM RESONANCE
# ─────────────────────────────────────────────────────────────────────────────

def run_hardware_experiment(
    distance: int    = 3,
    rounds: int      = 1,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "emerald",
    no_reset = False
) -> tuple[np.ndarray, stim.Circuit]:
    """
    Runs the surface code stabilizer circuit on IQM emerald via IQM Resonance.

    Follows the same connection pattern as emeraldRepetitionCode.ipynb:
        IQMProvider → get_backend → transpile → run → get_counts

    Parameters
    ----------
    distance         : code distance (use 3 for emerald)
    rounds           : measurement rounds (use 1 for NISQ hardware)
    shots            : number of circuit executions
    token            : IQM Resonance API token (prompts if None)
    quantum_computer : "emerald" or "emerald"
    emerald_origin    : (col, row) placement of the code on the emerald grid

    Returns
    -------
    raw_meas   : np.ndarray  shape (shots, num_measurements), dtype bool
                 Rows are shots; columns are in Stim's measurement order.
                 Ready to pass directly to stim's m2d converter.
    stim_circ  : stim.Circuit  noiseless circuit used for hardware.
                 Needed for detection event conversion.
    """
    from iqm.qiskit_iqm import IQMProvider

    # Build circuits, exchange for qiskit if desired
    stim_circ  = make_stim_circuit(distance, rounds, noise=None) # ADD NO RESET FLAG IF DESIRED
    qc, stim_to_dense, meas_order = stim_to_qiskit(stim_circ)

    # OPTIONAL: IF YOU WANT TO ASSIGN QUBIT MAP BY HAND 
    # Map abstract Qiskit qubits to emerald physical qubits
    qubit_map  = build_emerald_qubit_map_optimizing(stim_circ)
    
    # HIGHLY RECOMMEND PRINTING THIS, if you are using it.
    print(qubit_map)
    # initial_layout: Qiskit needs a list where position i = physical qubit for logical i
    n_qubits   = qc.num_qubits
    layout     = [qubit_map.get(i, i) for i in range(n_qubits)]

    # Connect to IQM Resonance
    provider   = IQMProvider(
        "https://resonance.meetiqm.com",
        quantum_computer=quantum_computer,
        token=get_token() 
    )
    backend    = provider.get_backend()

    # Transpile: Qiskit will insert SWAPs for diagonal CX pairs automatically
    qc_t       = transpile(qc, backend, 
                           initial_layout=layout, #IF YOU HAVE AN EMERALD QUBIT MAP AVAILABE, also change optimization_level=1 or 0
                            optimization_level=1)
    print(f"Transpiled circuit depth: {qc_t.depth()}  "
          f"(ideal: {qc.depth()}, extra depth from SWAPs)")
    print(f"Tanspiled circuit ops: {qc_t.count_ops()}  "
          f"(original {qc.count_ops()})")
    
    qc_t.draw(output="mpl", filename="transpiled.png")
    
    return None, None

    # Run
    job        = backend.run(qc_t, shots=shots)
    counts     = job.result().get_counts()

    # Convert counts dict → (shots, num_measurements) numpy array
    raw_meas   = counts_to_measurement_array(counts, stim_circ.num_measurements,
                                               shots)
    return raw_meas, stim_circ



def decode_hardware_results(
    syndromes: dict
) -> tuple[float, float]:
    """
    Decodes the output of extract_syndromes() via your desired decoder.

    NOTE: Other parameters can be important for this function, such as a noise model. 


    Parameters
    ----------
    syndromes           : dict  output of extract_syndromes()

    Returns
    -------
    ler : float   estimated logical error rate
    err : float   OPTIONAL 1-sigma statistical error
    """

    # USE e.g. PyMatching to decode

    ler,err = [], []
    return ler, err

