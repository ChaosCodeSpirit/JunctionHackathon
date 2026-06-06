

from internal_helpers import get_qubit_lists
import stim
from typing import Callable, Iterable, Optional

try:
    from iqm.qiskit_iqm import IQMProvider  # only needed for get_emerald_fidelities
except ImportError:
    IQMProvider = None



"""

OPTIONAL FILE: Allows you to build a custom STIM -> EMERALD mapping. 

"""


def _read_token(path: str = "TOKEN.txt") -> str:
    with open(path) as f:
        return f.readline().strip()


def get_emerald_fidelities(token: Optional[str] = None) -> tuple[dict, dict]:
    """
    Pull single-qubit (PRX) and two-qubit (CZ) gate fidelities from
    IQM Resonance for the Emerald backend.

    Returns
    -------
    qubit_fidelities   : dict  "QB<n>" -> PRX fidelity
    coupler_fidelities : dict  locus-tuple -> CZ fidelity
                         For Emerald (star topology) the loci are
                         ("COMP_R", "QB<n>"); for direct-coupled chips they
                         are ("QB<a>", "QB<b>").
    """
    if IQMProvider is None:
        raise ImportError("iqm.qiskit_iqm is not installed; cannot fetch fidelities.")
    provider   = IQMProvider(
        "https://resonance.meetiqm.com",
        quantum_computer="emerald",
        token=token if token is not None else _read_token(),
    )
    backend    = provider.get_backend(use_metrics=True)

    qubit_fidelities = {}
    coupler_fidelities = {}

    if backend.metrics is not None:
        for q in backend.architecture.qubits:
            locus = (q,)
            impl = backend.architecture.gates["prx"].get_default_implementation(locus)
            qubit_fidelities[q] = backend.metrics.get_gate_fidelity("prx", impl, locus)
        for locus in backend.architecture.gates["cz"].loci:
            impl = backend.architecture.gates["cz"].get_default_implementation(locus)
            coupler_fidelities[locus] = backend.metrics.get_gate_fidelity("cz", impl, locus)

    return qubit_fidelities, coupler_fidelities


def coordinate_to_emerald_qubit(
    coord: tuple,
    grid_top_left: int = 41,
    grid_n_rows: int = 5,
    grid_n_cols: int = 5,
    row_stride: int = -8,
    col_stride: int = 1,
    qiskit_offset: int = -1,
    off_r: int = 4,
    off_c: int = -2,
) -> int:
    """
    Translate a Stim grid coordinate (x, y) into the Qiskit / Resonance
    qubit index on the Emerald chip.

    The Stim rotated-surface-code coordinate system places data qubits on
    (odd, odd) positions and ancillas on (even, even) positions of a diamond
    lattice. CX partners are always *diagonal* neighbours, i.e. (delta x, delta y) = (+/-1, +/-1).
    Rotating the lattice by 45 degrees turns those diagonals into ordinary
    axis neighbours on a square grid:

        row = (y - x + off_r) / 2
        col = (x + y + off_c) / 2

    The default off_r=4, off_c=-2 are correct for distance-3; they shift the
    smallest (y-x) and smallest (x+y) values found in the d=3 layout to 0
    so the patch fits in rows 0..4, cols 0..4.

    Returns the Qiskit physical index (= Resonance QB number + qiskit_offset).
    """
    x, y = int(coord[0]), int(coord[1])
    if (x + y) % 2 != 0:
        raise ValueError(
            f"Stim coord {coord} has x+y odd; only diamond-lattice positions "
            f"(x+y even) are supported."
        )
    row = (y - x + off_r) // 2
    col = (x + y + off_c) // 2
    if not (0 <= row < grid_n_rows and 0 <= col < grid_n_cols):
        raise ValueError(
            f"Stim coord {coord} rotates to grid ({row},{col}) which is "
            f"outside the {grid_n_rows}x{grid_n_cols} sub-region."
        )
    qb = grid_top_left + row * row_stride + col * col_stride
    return qb + qiskit_offset


# -----------------------------------------------------------------------------
#  Shared helpers
# -----------------------------------------------------------------------------

def _stim_grid_positions(
    stim_circuit: stim.Circuit,
) -> tuple[dict, list, dict]:
    """
    Canonical 45-degree rotation of the Stim diamond lattice onto a square
    grid. Returns:
        positions     : dict  stim_q -> (row, col) in the un-transformed patch
        all_stim_q    : list  Stim qubit indices used by the circuit (sorted)
        stim_to_dense : dict  stim_q -> dense Qiskit index 0..N-1
    """
    coords = stim_circuit.get_final_qubit_coordinates()
    data_q, anc_q = get_qubit_lists(stim_circuit)
    all_stim_q = sorted(set(data_q + anc_q))
    stim_to_dense = {sq: i for i, sq in enumerate(all_stim_q)}

    xs = [int(coords[q][0]) for q in all_stim_q]
    ys = [int(coords[q][1]) for q in all_stim_q]
    off_r = -min(y - x for x, y in zip(xs, ys))
    off_c = -min(x + y for x, y in zip(xs, ys))

    positions = {}
    for sq in all_stim_q:
        x, y = int(coords[sq][0]), int(coords[sq][1])
        positions[sq] = ((y - x + off_r) // 2, (x + y + off_c) // 2)
    return positions, all_stim_q, stim_to_dense


def _dihedral_transforms(n_rows: int, n_cols: int) -> list[tuple[str, Callable, int, int]]:
    """
    Symmetries of a rectangular patch as (label, transform, n_rows_after,
    n_cols_after). All 8 D4 elements when n_rows == n_cols, otherwise the
    4 that preserve aspect ratio.
    """
    R, C = n_rows, n_cols
    base = [
        ("id",     lambda r, c, R=R, C=C: (r, c),               R, C),
        ("rot180", lambda r, c, R=R, C=C: (R - 1 - r, C - 1 - c), R, C),
        ("flipH",  lambda r, c, R=R, C=C: (r, C - 1 - c),         R, C),
        ("flipV",  lambda r, c, R=R, C=C: (R - 1 - r, c),         R, C),
    ]
    if R == C:
        base += [
            ("rot90",     lambda r, c, R=R, C=C: (c, R - 1 - r),           C, R),
            ("rot270",    lambda r, c, R=R, C=C: (C - 1 - c, r),           C, R),
            ("transpose", lambda r, c, R=R, C=C: (c, r),                   C, R),
            ("antitrans", lambda r, c, R=R, C=C: (C - 1 - c, R - 1 - r),   C, R),
        ]
    return base


def _unique_cx_pairs(stim_circuit: stim.Circuit, stim_to_dense: dict) -> set[frozenset]:
    """Set of unordered Stim-qubit pairs that share a CX/CZ in the circuit."""
    pairs: set[frozenset] = set()
    for instr in stim_circuit.flattened():
        if instr.name not in ("CX", "CZ"):
            continue
        targets = [t.value for t in instr.targets_copy() if t.is_qubit_target]
        for i in range(0, len(targets), 2):
            a, b = targets[i], targets[i + 1]
            if a in stim_to_dense and b in stim_to_dense:
                pairs.add(frozenset((a, b)))
    return pairs


def _edge_fidelity(
    coupler_fidelities: dict,
    qa_name: str,
    qb_name: str,
    resonator_name: str = "COMP_R",
) -> Optional[float]:
    """
    Look up a 2-qubit gate fidelity, accepting either:
      - direct edge keys  (QBa, QBb) or (QBb, QBa)         [Garnet style]
      - resonator-mediated: product of (R, QBa) and (R, QBb)
        in either ordering                                  [Emerald style]
    Returns None if the edge is not natively supported.
    """
    for k in ((qa_name, qb_name), (qb_name, qa_name)):
        if k in coupler_fidelities:
            return coupler_fidelities[k]

    def _res_link(qb):
        return (
            coupler_fidelities.get((resonator_name, qb))
            or coupler_fidelities.get((qb, resonator_name))
        )

    fa, fb = _res_link(qa_name), _res_link(qb_name)
    if fa is not None and fb is not None:
        return fa * fb
    return None


# -----------------------------------------------------------------------------
#  Optimizing placement
# -----------------------------------------------------------------------------

def build_emerald_qubit_map_optimizing(
    stim_circuit: stim.Circuit,
    qubit_fidelities: Optional[dict] = None,
    coupler_fidelities: Optional[dict] = None,
    grid_n_rows: int = 5,
    grid_n_cols: int = 5,
    row_stride: int = -8,
    col_stride: int = 1,
    qiskit_offset: int = -1,
    candidate_top_lefts: Optional[Iterable[int]] = None,
    try_orientations: bool = True,
    resonator_name: str = "COMP_R",
    token: Optional[str] = None,
    verbose: bool = False,
) -> tuple[dict, float, dict]:
    """
    Choose the best valid location on the Emerald qubit grid for the surface
    code patch (no SWAPs needed) by *maximizing*

        score = prod(qubit_fidelity[q] for q in qubits_used)
              * prod(edge_fidelity[(a,b)] for (a,b) in CX_pairs_used)

    The search space is every (grid_top_left, orientation) such that:
      - all 17 patch positions land on real chip qubits, and
      - every CX pair lands on a natively-coupled edge (or, for Emerald-style
        resonator chips, both endpoints couple to the same COMP_R).
    Orientations enumerated: the 8 dihedral symmetries of the patch (4 if the
    patch is rectangular).

    Parameters
    ----------
    stim_circuit        : the surface-code circuit
    qubit_fidelities    : "QB<n>" -> PRX fidelity. If None, fetched via
                          get_emerald_fidelities(token=token).
    coupler_fidelities  : locus-tuple -> CZ fidelity. If None, fetched.
    grid_n_rows,
    grid_n_cols         : patch shape on hardware (square for d=3).
    row_stride,
    col_stride          : QB-number step per (row, col) move on the chip.
    qiskit_offset       : Qiskit index = Resonance QB number + qiskit_offset
                          (Resonance is 1-indexed -> -1).
    candidate_top_lefts : QB numbers to try as the (row=0, col=0) anchor.
                          Defaults to every QB on the chip.
    try_orientations    : if False, only the canonical orientation is tried.
    resonator_name      : name of the computational resonator on the chip
                          (used for fallback edge lookup).
    verbose             : print every attempted placement and its score.

    Returns
    -------
    qubit_map : dict   dense_stim_index -> qiskit_physical_index
                       Ready to pass to transpile(initial_layout=...).
    score     : float  product of fidelities for the chosen placement (1 is perfect).
    info      : dict   diagnostic info:
                         "top_left"    -> chosen QB anchor
                         "orientation" -> symmetry label
                         "n_tried"     -> total placements considered
                         "n_valid"     -> placements that fit + are coupled
                         "qb_names"    -> set of physical QBs used
    """
    if qubit_fidelities is None or coupler_fidelities is None:
        qf, cf = get_emerald_fidelities(token=token)
        if qubit_fidelities is None:
            qubit_fidelities = qf
        if coupler_fidelities is None:
            coupler_fidelities = cf

    positions, _all_stim_q, stim_to_dense = _stim_grid_positions(stim_circuit)
    cx_pairs = _unique_cx_pairs(stim_circuit, stim_to_dense)

    if candidate_top_lefts is None:
        candidate_top_lefts = sorted(
            int(name[2:]) for name in qubit_fidelities
            if isinstance(name, str) and name.startswith("QB") and name[2:].isdigit()
        )

    transforms = (
        _dihedral_transforms(grid_n_rows, grid_n_cols)
        if try_orientations
        else [_dihedral_transforms(grid_n_rows, grid_n_cols)[0]]
    )

    best = None
    n_tried = 0
    n_valid = 0

    for tl in candidate_top_lefts:
        for label, transform, n_r_after, n_c_after in transforms:
            n_tried += 1

            # 1) Build placement -> Qiskit indices
            qmap: dict[int, int] = {}
            qb_names_used: set[str] = set()
            placement_ok = True
            for sq, (r, c) in positions.items():
                r_t, c_t = transform(r, c)
                if not (0 <= r_t < n_r_after and 0 <= c_t < n_c_after):
                    placement_ok = False
                    break
                qb = tl + r_t * row_stride + c_t * col_stride
                name = f"QB{qb}"
                if name not in qubit_fidelities:
                    placement_ok = False
                    break
                qmap[stim_to_dense[sq]] = qb + qiskit_offset
                qb_names_used.add(name)
            if not placement_ok:
                continue

            # 2) Score qubits and edges
            score = 1.0
            for name in qb_names_used:
                score *= qubit_fidelities[name]

            edges_ok = True
            for pair in cx_pairs:
                sa, sb = tuple(pair)
                qa = qmap[stim_to_dense[sa]]
                qb = qmap[stim_to_dense[sb]]
                ef = _edge_fidelity(
                    coupler_fidelities,
                    f"QB{qa - qiskit_offset}",
                    f"QB{qb - qiskit_offset}",
                    resonator_name=resonator_name,
                )
                if ef is None:
                    edges_ok = False
                    break
                score *= ef
            if not edges_ok:
                continue

            n_valid += 1
            if verbose:
                print(f"  tl=QB{tl:<2}  orient={label:<9}  score={score:.6f}")

            if best is None or score > best["score"]:
                best = dict(
                    qubit_map=dict(qmap),
                    score=score,
                    top_left=tl,
                    orientation=label,
                    qb_names=set(qb_names_used),
                )

    if best is None:
        raise RuntimeError(
            f"No valid placement found (tried {n_tried}). "
            f"Check qubit_fidelities/coupler_fidelities coverage and grid "
            f"parameters."
        )

    info = {
        "top_left": best["top_left"],
        "orientation": best["orientation"],
        "n_tried": n_tried,
        "n_valid": n_valid,
        "qb_names": best["qb_names"],
    }
    return best["qubit_map"], best["score"], info


def build_emerald_qubit_map(
    stim_circuit: stim.Circuit,
    grid_top_left: int = 41,
    grid_n_rows: int = 5,
    grid_n_cols: int = 5,
    row_stride: int = -8,
    col_stride: int = 1,
    qiskit_offset: int = -1,
) -> dict:
    """
    Maps Stim dense qubit indices (0..16) to Qiskit qubit indices for IQM Emerald.

    NOTE: OPTIONAL, since the qiskit transpiler automatically optimizes for least number of swap gates. Still, hand-testing is recommended.

    Default parameters target this 5x5 sub-region of the Emerald chip (OR ANY OTHER!)

        QB41 QB42 QB43 QB44 QB45   <- grid row 0 (top)
        QB33 QB34 QB35 QB36 QB37
        QB25 QB26 QB27 QB28 QB29
        QB17 QB18 QB19 QB20 QB21
        QB9  QB10 QB11 QB12 QB13   <- grid row 4 (bottom)

    This maps the code's natural diagonal connectivity directly onto the hardware grid, giving
    ALL 24 stabilizer CX pairs a Manhattan distance of 1 - no SWAPs needed.

    EXAMPLARY Resulting layout on the 5x5 hardware sub-region:

        col:   0     1     2     3     4
        row 0:  .     .   D(5)  A(13)  .       QB41-QB45
        row 1: A(2)  D(3)  A(11) D(12)  .       QB33-QB37
        row 2: D(1)  A(9)  D(10) A(18) D(19)    QB25-QB29
        row 3:  .    D(8)  A(16) D(17) A(25)    QB17-QB21
        row 4:  .    A(14) D(15)  .     .       QB9 -QB13

    (numbers are Stim qubit indices; 8 spare positions at edges)

    IQM Resonance uses 1-based QB numbers (QB1, QB2, ...).
    Qiskit uses 0-based indices.
    The output could be ready ready to pass directly to transpile(initial_layout=...)

    Parameters
    ----------
    stim_circuit  : stim.Circuit produced by surface_code:rotated_memory_{x,z}
    grid_top_left : Resonance QB number at (row=0, col=0). Default 41.
    grid_n_rows   : rows in the sub-region. Default 5.
    grid_n_cols   : cols in the sub-region. Default 5.
    row_stride    : QB number delta per row step downward. Default -8.
    col_stride    : QB number delta per col step rightward. Default 1.
    qiskit_offset : added to QB numbers to get Qiskit 0-based indices. Default -1.

    Returns
    -------
    dict  dense_stim_index -> qiskit_physical_index
          Ready for: transpile(qc, backend, initial_layout=qubit_map)

    Notes
    -----
    Verify before hardware runs by cross-checking with backend.coupling_map.
    THIS NEEDS A FULL REWORK, if you want to use it for the Ising model, i.e. it needs to based on the Ising Models MemoryCiruict functionality, see github.com/NVIDIA/Ising-Decoding/code/qec/surface_code/memory_circuit.py
    """
    coords        = stim_circuit.get_final_qubit_coordinates()
    data_q, anc_q = get_qubit_lists(stim_circuit)
    all_stim_q    = sorted(set(data_q + anc_q))
    stim_to_dense = {sq: i for i, sq in enumerate(all_stim_q)}

    # 45-degree rotation of the diamond lattice onto a square grid.
    #   row = (y - x + off_r) / 2,  col = (x + y + off_c) / 2
    # off_r, off_c shift the smallest (y-x) and (x+y) values used by the
    # circuit to 0 so the patch always starts at grid (0, 0).
    xs = [int(coords[q][0]) for q in all_stim_q]
    ys = [int(coords[q][1]) for q in all_stim_q]
    off_r = -min(y - x for x, y in zip(xs, ys))
    off_c = -min(x + y for x, y in zip(xs, ys))

    qubit_map = {}
    for sq in all_stim_q:
        x, y = int(coords[sq][0]), int(coords[sq][1])
        row = (y - x + off_r) // 2
        col = (x + y + off_c) // 2
        if not (0 <= row < grid_n_rows and 0 <= col < grid_n_cols):
            raise ValueError(
                f"Stim qubit {sq} at ({x},{y}) rotates to grid ({row},{col}) "
                f"which is outside the {grid_n_rows}x{grid_n_cols} sub-region. "
                f"Increase grid_n_rows/grid_n_cols or pick a larger top-left."
            )
        qb = grid_top_left + row * row_stride + col * col_stride
        qubit_map[stim_to_dense[sq]] = qb + qiskit_offset

    return qubit_map





def print_qubit_map(
    stim_circuit: stim.Circuit,
    grid_top_left: int = 41,
    grid_n_rows: int = 5,
    grid_n_cols: int = 5,
    row_stride: int = -8,
    col_stride: int = 1,
    qiskit_offset: int = -1,
):
    """
    NOTE: If you are using qiskit, the mapping is easily compared by hand, since the expected Qiskit and Resonance qubit indices are the same (except for a -1 difference).


    Prints a human-readable table of the Stim → Emerald qubit mapping
    and flags any CX pairs that require SWAPs (non-adjacent physical qubits).
    Use this to verify the mapping before submitting to hardware.

    ONLY WORKS ON SQUARE PATCHES
    

    Parameters
    ----------
    stim_circuit   : stim.Circuit
    grid_top_left  : Resonance QB number at (row=0, col=0). Default 41.
    grid_n_rows    : number of rows in the sub-region. Default 5.
    grid_n_cols    : number of columns in the sub-region. Default 5.
    row_stride     : QB number difference per row step downward. Default -8. I.e. row 0 starts at QB41, row 1 starts QB33
    col_stride     : QB number difference per column step rightward. Default 1.
    qiskit_offset  : added to QB numbers to get Qiskit 0-based indices.
                     Default -1 (QB1 -> 0, QB17 -> 16, QB45 -> 44).
                     Set to 0 to return raw Resonance QB numbers instead.
    """
    coords        = stim_circuit.get_final_qubit_coordinates()
    data_q, anc_q = get_qubit_lists(stim_circuit)
    qubit_map     = build_emerald_qubit_map(
        stim_circuit, grid_top_left, grid_n_rows, grid_n_cols,
        row_stride, col_stride, qiskit_offset,
    )
    all_stim_q    = sorted(set(data_q + anc_q))
    stim_to_dense = {sq: i for i, sq in enumerate(all_stim_q)}
 
    print("─" * 68)
    print(f"{'Stim idx':>10} {'Stim coord':>12} {'Type':>8} "
          f"{'Dense idx':>10} {'Qiskit idx':>12} {'QB (Resonance)':>16}")
    print("─" * 68)
    for stim_q in all_stim_q:
        d    = stim_to_dense[stim_q]
        qk   = qubit_map[d]
        xy   = coords[stim_q]
        qt   = "DATA " if stim_q in set(data_q) else "ANCL "
        print(f"{stim_q:>10} {str((int(xy[0]),int(xy[1]))):>12} "
              f"{qt:>8} {d:>10} {qk:>12} {'QB'+str(qk-qiskit_offset):>16}")
    print("─" * 68)
 
    def qiskit_to_grid(q):
        """Convert Qiskit index to (row, col) in the Emerald sub-grid.
        Derived from: q = (grid_top_left + qiskit_offset) + r*row_stride + c*col_stride
        For default params: q = 40 + r*(-8) + c*1  →  r = (40-q+q%8)//8, c = q%8
        General: solves for r and c given the stride parameters."""
        # Shift to remove qiskit_offset: get QB number back, then find grid pos
        qb = q - qiskit_offset          # Resonance QB number
        # qb = grid_top_left + r*row_stride + c*col_stride
        # Since col_stride=1: c = (qb - grid_top_left - r*row_stride)
        # Iterate rows (small loop, only grid_n_rows iterations)
        for r in range(grid_n_rows):
            c = qb - grid_top_left - r * row_stride
            if 0 <= c < grid_n_cols:
                return r, c
        return None, None  # not in this sub-grid
 
    # CX connectivity check
    n_native, n_swap = 0, 0
    swap_details = []
    for instr in stim_circuit.flattened():
        if instr.name == "CX":
            targets = [t.value for t in instr.targets_copy() if t.is_qubit_target]
            for i in range(0, len(targets), 2):
                sa, sb = targets[i], targets[i + 1]
                qa = qubit_map[stim_to_dense[sa]]
                qb = qubit_map[stim_to_dense[sb]]
                ra, ca = qiskit_to_grid(qa)
                rb, cb = qiskit_to_grid(qb)
                dist = abs(ra - rb) + abs(ca - cb)
                if dist == 1:
                    n_native += 1
                else:
                    n_swap += 1
                    swap_details.append((sa, sb, qa, qb, dist))
 
    print(f"CX gates: {n_native} native (dist=1),  "
          f"{n_swap} non-native (dist>1, need SWAP)")
    if swap_details:
        print("  Non-native pairs (sorted by distance):")
        for sa, sb, qa, qb, d in sorted(swap_details, key=lambda x: -x[4]):
            print(f"    stim({sa:>2},{sb:>2})  QB{qa-qiskit_offset:>2}↔QB{qb-qiskit_offset:>2}"
                  f"  Qiskit({qa},{qb})  dist={d}")
    print()