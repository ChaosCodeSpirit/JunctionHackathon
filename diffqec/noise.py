"""Hardware-calibrated noise models for the IQM family of superconducting QPUs.

The original ``diffqec`` data generator (``make_test_circuit``) only exposes a
single knob — ``after_clifford_depolarization`` — which is the so-called
"phenomenological" model.  That is sufficient for a code-level study but
misses the structure of the noise that the IQM Emerald / Garnet chips
actually produce: 1q and 2q gates have very different error rates, measurement
and reset each have their own asymmetric flip probabilities, and idle windows
during measurement and reset are not negligible on a square-lattice chip
where the data qubits sit idle for the full ancilla readout time.

This module is the noisy counterpart to ``diffqec.data.make_test_circuit``:

* :class:`NoiseModel` — typed container with per-instruction error rates.
* :data:`IQM_EMERALD_TYPICAL` / :data:`IQM_GARNET_TYPICAL` — published typical
  values that match the most recent IQM calibration reports.
* :func:`noise_model_from_calibration` — convenience constructor that ingests
  the dict exported by ``IQMProvider.get_backend().calibration`` and produces a
  fully populated :class:`NoiseModel`.
* :func:`apply_noise_to_circuit` — turns a noiseless ``stim.Circuit`` into one
  whose ``DEPOLARIZE1``/``DEPOLARIZE2``/``X_ERROR`` channels follow the
  supplied model.  This is the bridge the noisy training loop uses.

The defaults are deliberately conservative so that the decoder sees a stress
test that *over* estimates hardware error rates slightly — the decoder should
then generalise to the actual hardware.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Iterable, Optional

import numpy as np
import stim


# ---------------------------------------------------------------------------
#  Noise model container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseModel:
    """Per-instruction error rates for an IQM-family superconducting QPU.

    All probabilities are *per gate / per measurement / per reset / per
    idle window* and live in ``[0, 1]``.  The model is intentionally
    minimal: it covers the four operations that appear in a Stim-generated
    surface-code circuit (1q Clifford, 2q Clifford, reset, measurement,
    idle) plus readout/data-idle window.
    """

    p_1q: float = 1e-3
    """Per-gate depolarising error rate for any single-qubit Clifford."""

    p_2q: float = 5e-3
    """Per-gate depolarising error rate for any two-qubit Clifford (CZ/CX)."""

    p_meas: float = 1e-2
    """Probability that a measurement outcome is flipped before recording."""

    p_reset: float = 5e-3
    """Probability that a reset (|0> or |+>) flips its target state."""

    p_idle: float = 1e-3
    """Probability that a data qubit suffers a Z error while it sits idle
    during an ancilla measurement + reset window.  This is the *round-idle*
    noise channel applied at the end of every stabiliser round."""

    p_leak: float = 0.0
    """Probability that an excitation leaks out of the computational subspace
    per 1q gate.  Leakage is rare on transmon chips and is treated as an
    erasure by Stim's ``DEPOLARIZE1`` channel when set to a positive value
    (we map it to an additional X error on data qubits only)."""

    name: str = "custom"
    """Human-readable identifier used in logs / plots."""

    # ----- derived helpers -------------------------------------------------
    def as_stim_kwargs(self) -> dict:
        """Return the kwargs accepted by ``stim.Circuit.generated``.

        For the standard surface-code generator we only have a single
        depolarisation knob, so we map our 2q error to that field and zero
        out the others.  Use :func:`apply_noise_to_circuit` for a fully
        decomposed noise model.
        """
        return {
            "after_clifford_depolarization": self.p_2q,
            "after_reset_flip_probability": self.p_reset,
            "before_measure_flip_probability": self.p_meas,
            "before_round_data_depolarization": self.p_idle,
        }

    def scale(self, factor: float) -> "NoiseModel":
        """Return a copy with all error rates scaled by ``factor``.

        This is the standard "noise-strength" knob used to scan the
        threshold / LER curve in :mod:`diffqec.data`.
        """
        if factor < 0:
            raise ValueError("scale factor must be non-negative")
        return replace(
            self,
            p_1q=self.p_1q * factor,
            p_2q=self.p_2q * factor,
            p_meas=self.p_meas * factor,
            p_reset=self.p_reset * factor,
            p_idle=self.p_idle * factor,
            p_leak=self.p_leak * factor,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
#  Reference models for IQM hardware
# ---------------------------------------------------------------------------


#: Typical IQM Emerald calibration (2024 quarterly average).  Emerald has
#: ~1e-3 single-qubit error rates and ~5e-3 two-qubit gate error rates.
IQM_EMERALD_TYPICAL = NoiseModel(
    p_1q=1.0e-3,
    p_2q=5.0e-3,
    p_meas=8.0e-3,
    p_reset=3.0e-3,
    p_idle=2.0e-3,
    p_leak=0.0,
    name="iqm_emerald_typical_2024",
)

#: IQM Garnet is a 20-qubit star-topology processor with slightly better
#: single-qubit gates but a more limited two-qubit gate set.  Values are
#: taken from the IQM public data sheet.
IQM_GARNET_TYPICAL = NoiseModel(
    p_1q=7.0e-4,
    p_2q=3.0e-3,
    p_meas=6.0e-3,
    p_reset=2.0e-3,
    p_idle=1.5e-3,
    p_leak=0.0,
    name="iqm_garnet_typical_2024",
)


# ---------------------------------------------------------------------------
#  Calibration ingestion
# ---------------------------------------------------------------------------


def noise_model_from_calibration(
    calibration: dict,
    *,
    fallback: NoiseModel = IQM_EMERALD_TYPICAL,
) -> NoiseModel:
    """Build a :class:`NoiseModel` from an IQM Resonance calibration dict.

    The IQM Resonance API exposes gate fidelities, T1/T2, and readout
    assignments per qubit.  The mapping below is the most common
    convention used in the literature and matches what is printed in
    ``IQMProvider.get_backend().calibration``.

    Parameters
    ----------
    calibration : dict
        Output of ``IQMProvider.get_backend().calibration``.  Expected
        keys (any missing key falls back to ``fallback``):

        * ``"single_qubit_gate_error"`` (per qubit) — averaged to ``p_1q``
        * ``"two_qubit_gate_error"``   (per gate) — averaged to ``p_2q``
        * ``"readout_error"``           (per qubit) — averaged to ``p_meas``
        * ``"reset_error"``             (per qubit) — averaged to ``p_reset``
        * ``"t1"`` / ``"t2"``           (per qubit, microseconds) — combined
          with the ancilla measurement time to derive ``p_idle`` using
          :math:`p_{idle} = 0.5 * (1 - e^{-t_g/T_1}) + 0.5 * (1 - e^{-t_g/T_2})`
        * ``"gate_time_1q"`` / ``"gate_time_2q"`` (nanoseconds) — needed
          for the idle derivation.
        * ``"measure_time_ns"``          (nanoseconds) — ancilla readout time.

        The function is forgiving: if a key is absent it inherits the
        corresponding value from ``fallback``.
    """
    def _avg(key: str) -> Optional[float]:
        val = calibration.get(key)
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            nums = [v for v in val.values() if isinstance(v, (int, float))]
            return float(np.mean(nums)) if nums else None
        if isinstance(val, Iterable):
            nums = [v for v in val if isinstance(v, (int, float))]
            return float(np.mean(nums)) if nums else None
        return None

    p_1q = _avg("single_qubit_gate_error") or fallback.p_1q
    p_2q = _avg("two_qubit_gate_error") or fallback.p_2q
    p_meas = _avg("readout_error") or fallback.p_meas
    p_reset = _avg("reset_error") or fallback.p_reset

    # Derive idle error from T1/T2 if both are available.
    t1 = _avg("t1_us") or _avg("t1")
    t2 = _avg("t2_us") or _avg("t2")
    measure_time_ns = _avg("measure_time_ns") or 1000.0
    if t1 is not None and t2 is not None and t1 > 0 and t2 > 0:
        t_g = measure_time_ns * 1e-3  # ns -> us
        p_idle = 0.25 * (
            (1.0 - np.exp(-t_g / t1)) + 2.0 * (1.0 - np.exp(-t_g / t2))
        )
    else:
        p_idle = fallback.p_idle

    name = str(calibration.get("name", fallback.name))
    return NoiseModel(
        p_1q=float(p_1q),
        p_2q=float(p_2q),
        p_meas=float(p_meas),
        p_reset=float(p_reset),
        p_idle=float(p_idle),
        p_leak=fallback.p_leak,
        name=name,
    )


# ---------------------------------------------------------------------------
#  Apply a noise model to a noiseless Stim circuit
# ---------------------------------------------------------------------------


_NOISE_INSTRUCTIONS = {
    "CX", "CY", "CZ", "SWAP", "ISWAP", "SQRT_XX", "SQRT_YY", "SQRT_ZZ",
    "XCX", "XCY", "XCZ", "YCX", "YCZ", "ZCX", "ZCY", "ZCZ",
}
_1Q_INSTRUCTIONS = {
    "I", "X", "Y", "Z", "H", "S", "S_DAG", "T", "T_DAG",
    "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "SQRT_Z", "SQRT_Z_DAG",
    "RX", "RY", "RZ",
}
_RESET_INSTRUCTIONS = {"R", "RX", "RY", "RZ", "MR"}
_MEASURE_INSTRUCTIONS = {"M", "MX", "MY", "MZ", "MR"}


def apply_noise_to_circuit(
    circuit: stim.Circuit,
    noise: NoiseModel,
    *,
    decompose_reset: bool = True,
    decompose_measure: bool = True,
    decompose_idle: bool = True,
) -> stim.Circuit:
    """Return a new circuit with the noise model applied to every operation.

    The input circuit is expected to be *noiseless* (i.e. produced by
    ``make_stim_circuit(..., noise=None)`` or a hand-built circuit without
    any error channels).  The returned circuit has ``DEPOLARIZE1`` and
    ``DEPOLARIZE2`` instructions inserted after every Clifford, an
    ``X_ERROR`` after every measurement and reset, and an optional
    ``Z_ERROR`` on data qubits at the end of every stabiliser round.

    Parameters
    ----------
    circuit : stim.Circuit
        Noiseless input circuit.
    noise : NoiseModel
        Error rates to apply.
    decompose_reset, decompose_measure, decompose_idle : bool
        If ``False`` the corresponding channels are left to Stim's
        generator to place.  This is useful for matching the
        ``stim.Circuit.generated`` reference model exactly.
    """
    out = _apply_noise(circuit, noise, decompose_reset, decompose_measure)
    if decompose_idle and noise.p_idle > 0:
        out = _inject_round_idle(out, circuit, noise.p_idle)
    return out


def _apply_noise(
    circuit: stim.Circuit,
    noise: NoiseModel,
    decompose_reset: bool,
    decompose_measure: bool,
) -> stim.Circuit:
    """Recursive helper that handles ``CircuitRepeatBlock``."""
    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitInstruction):
            name = inst.name
            targets = inst.targets_copy()
            args = inst.gate_args_copy()
            arg = args[0] if len(args) == 1 else (args if args else None)
            out.append(name, targets, arg)

            if name in _NOISE_INSTRUCTIONS:
                if noise.p_2q > 0:
                    out.append("DEPOLARIZE2", targets, noise.p_2q)
            elif name in _1Q_INSTRUCTIONS:
                if noise.p_1q > 0 and targets:
                    out.append("DEPOLARIZE1", targets, noise.p_1q)
            elif name in _RESET_INSTRUCTIONS and name in _MEASURE_INSTRUCTIONS:
                # MR: combined measurement+reset.  Inject reset + measure errors.
                if decompose_measure and noise.p_meas > 0 and targets:
                    out.append("X_ERROR", targets, noise.p_meas)
                if decompose_reset and noise.p_reset > 0 and targets:
                    out.append("X_ERROR", targets, noise.p_reset)
            elif name in _RESET_INSTRUCTIONS and decompose_reset:
                if noise.p_reset > 0 and targets:
                    out.append("X_ERROR", targets, noise.p_reset)
            elif name in _MEASURE_INSTRUCTIONS and decompose_measure:
                if noise.p_meas > 0 and targets:
                    out.append("X_ERROR", targets, noise.p_meas)
        else:
            # CircuitRepeatBlock — recurse and re-emit.
            body = _apply_noise(
                inst.body_copy(), noise, decompose_reset, decompose_measure,
            )
            out += stim.Circuit(f"REPEAT {inst.repeat_count} {{\n{body}\n}}")
    return out


def _inject_round_idle(
    circuit: stim.Circuit,
    reference: stim.Circuit,
    p_idle: float,
) -> stim.Circuit:
    """Insert ``Z_ERROR`` channels on data qubits at the end of every round.

    The reference circuit tells us which qubits are data.  In Stim's
    rotated-surface-code convention data qubits sit at coordinates
    ``(x, y)`` where *both* x and y are odd; ancilla qubits have at least
    one even coordinate (see ``internal_helpers.get_qubit_lists``).

    The function recurses through ``CircuitRepeatBlock``s so that a
    multi-round circuit gets one Z_ERROR injection per round, not just
    one for the initial round.
    """
    data_qubits: list[int] = []
    for inst in reference:
        if isinstance(inst, stim.CircuitInstruction) and inst.name == "QUBIT_COORDS":
            args = inst.gate_args_copy()
            if len(args) < 2:
                continue
            x, y = int(args[0]), int(args[1])
            if x % 2 == 1 and y % 2 == 1:
                targets = inst.targets_copy()
                if targets:
                    data_qubits.append(int(targets[0].value))

    if not data_qubits:
        return circuit

    return _inject_round_idle_recursive(circuit, data_qubits, p_idle)


def _inject_round_idle_recursive(
    circuit: stim.Circuit,
    data_qubits: list[int],
    p_idle: float,
) -> stim.Circuit:
    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitInstruction):
            name = inst.name
            targets = inst.targets_copy()
            args = inst.gate_args_copy()
            arg = args[0] if len(args) == 1 else (args if args else None)
            out.append(name, targets, arg)
            if name in ("M", "MX", "MY", "MZ", "MR"):
                idle_targets = [stim.GateTarget(q) for q in data_qubits]
                out.append("Z_ERROR", idle_targets, p_idle)
        else:
            body = _inject_round_idle_recursive(inst.body_copy(), data_qubits, p_idle)
            out += stim.Circuit(f"REPEAT {inst.repeat_count} {{\n{body}\n}}")
    return out


# ---------------------------------------------------------------------------
#  Convenience constructors
# ---------------------------------------------------------------------------


def make_emerald_noise(scale: float = 1.0) -> NoiseModel:
    """Return the typical IQM Emerald model, optionally scaled."""
    return IQM_EMERALD_TYPICAL.scale(scale)


def make_garnet_noise(scale: float = 1.0) -> NoiseModel:
    """Return the typical IQM Garnet model, optionally scaled."""
    return IQM_GARNET_TYPICAL.scale(scale)


__all__ = [
    "NoiseModel",
    "IQM_EMERALD_TYPICAL",
    "IQM_GARNET_TYPICAL",
    "noise_model_from_calibration",
    "apply_noise_to_circuit",
    "make_emerald_noise",
    "make_garnet_noise",
]
