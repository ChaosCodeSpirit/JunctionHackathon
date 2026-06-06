from surface_code import *
from build_emerald_qubit_rotated import *
from extract_syndromes import *


TOKEN = ""


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
    qubit_map  = build_emerald_qubit_map(stim_circ)
    
    # HIGHLY RECOMMEND PRINTING THIS, if you are using it.
    # print(qubit_map)
    # initial_layout: Qiskit needs a list where position i = physical qubit for logical i
    n_qubits   = qc.num_qubits
    layout     = [qubit_map.get(i, i) for i in range(n_qubits)]

    # Connect to IQM Resonance
    provider   = IQMProvider(
        "https://resonance.meetiqm.com",
        quantum_computer=quantum_computer,
        token=TOKEN 
    )
    backend    = provider.get_backend()

    # Transpile: Qiskit will insert SWAPs for diagonal CX pairs automatically
    qc_t       = transpile(qc, backend, 
                           #initial_layout=layout, #IF YOU HAVE AN EMERALD QUBIT MAP AVAILABE, also change optimization_level=1 or 0
                            optimization_level=3)
    print(f"Transpiled circuit depth: {qc_t.depth()}  "
          f"(ideal: {qc.depth()}, extra depth from SWAPs)")

    # Run
    job        = backend.run(qc_t, shots=shots)
    counts     = job.result().get_counts()

    # Convert counts dict → (shots, num_measurements) numpy array
    raw_meas   = counts_to_measurement_array(counts, stim_circ.num_measurements,
                                               shots)
    return raw_meas, stim_circ



def decode_hardware_results(
    syndromes: dict,
    decoder: str = "mwpm",
    model_path: str | None = None,
    syndrome_shape: tuple | None = None,
    L: int | None = None,
) -> tuple[list, list]:
    """
    Decodes the output of extract_syndromes() via your desired decoder.

    Parameters
    ----------
    syndromes      : dict  output of extract_syndromes()
    decoder        : "mwpm" (default) or "diffqec"
    model_path     : path to DiffQEC checkpoint (required if decoder="diffqec")
    syndrome_shape : (rounds, D) expected by DiffQEC model
    L              : number of logical observables for DiffQEC

    Returns
    -------
    ler : list[float]  — per-observable logical error rate
    err : list[float]  — 1-sigma statistical error per observable
    """
    if decoder == "diffqec":
        from diffqec.integrate import decode_hardware_results_diffqec
        return decode_hardware_results_diffqec(
            syndromes,
            model_path=model_path,
            syndrome_shape=syndrome_shape,
            L=L,
        )

    # Default / placeholder: MWPM or empty
    ler, err = [], []
    return ler, err


def make_noisy_hardware_circuit(
    distance: int = 3,
    rounds: int = 3,
    *,
    noise: "NoiseModel | None" = None,
    memory: str = "Z",
    no_reset: bool = False,
):
    """Build a *noisy* (but hardware-shaped) Stim circuit for offline training.

    Mirrors :func:`run_hardware_experiment` but returns the circuit that
    *should* have been executed on the QPU, with a realistic noise model
    applied.  Use this to (a) validate the decoder against a hardware-like
    DEM and (b) generate training data on LUMI when you do not have QPU
    time.

    Parameters
    ----------
    distance, rounds, memory, no_reset
        As in :func:`make_stim_circuit`.
    noise : NoiseModel, optional
        Defaults to :data:`IQM_EMERALD_TYPICAL` when ``None``.
    """
    from diffqec.data import make_noisy_circuit
    from diffqec.noise import IQM_EMERALD_TYPICAL
    return make_noisy_circuit(
        distance=distance, rounds=rounds,
        noise=noise or IQM_EMERALD_TYPICAL,
        memory=memory, no_reset=no_reset,
    )

