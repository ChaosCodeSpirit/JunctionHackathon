"""54-qubit surface-code benchmark for IQM Q50 / VTT QX-style benchmarks.

Builds and simulates a 54-qubit rotated surface code (d=5 + 5 ancilla),
with optional DiffQEC decoding vs trivial / MWPM baseline.
"""
from .circuit import SurfaceCode54Q, build_memory_z_circuit, N_QUBITS, DISTANCE
from .simulate import run_stabilizer_simulation
from .benchmark import main as benchmark_main
from .train_diffqec import train_diffqec

__all__ = [
    "SurfaceCode54Q",
    "build_memory_z_circuit",
    "run_stabilizer_simulation",
    "benchmark_main",
    "train_diffqec",
    "N_QUBITS",
    "DISTANCE",
]
