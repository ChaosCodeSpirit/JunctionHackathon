from surface_code import make_stim_circuit, stim_to_qiskit


def test_build_draw_circuit(
    distance: int = 3,
    rounds: int = 1,
    memory: str = "z",
    out_path: str = "surface_code_circuit.txt",
) -> None:
    """
    Build a (noiseless) Stim surface-code circuit, convert to Qiskit, and
    draw the result to a file.

    A noiseless circuit is used because noise/annotation instructions are
    dropped by stim_to_qiskit and would only clutter the rendered output.
    """
    stim_circuit = make_stim_circuit(
        distance=distance,
        rounds=rounds,
        memory=memory,
        noise={
            "after_clifford_depolarization": 0.0,
            "after_reset_flip_probability": 0.0,
            "before_measure_flip_probability": 0.0,
            "before_round_data_depolarization": 0.0,
        },
    )
    coords = stim_circuit.get_final_qubit_coordinates()
    print(f"Qubit coordinates in stim circuit: {coords}")

    qc, stim_to_dense, meas_order = stim_to_qiskit(stim_circuit)

    print(f"Qiskit circuit: {qc.num_qubits} qubits, depth {qc.depth()}, "
          f"{len(qc.cregs)} classical register(s), "
          f"{qc.size()} instruction(s)")

    if out_path.endswith(".png") or out_path.endswith(".pdf") or out_path.endswith(".svg"):
        qc.draw(output="mpl", filename=out_path)
        print(f"Matplotlib drawing written to: {out_path}")
    else:
        with open(out_path, "w") as f:
            f.write(qc.draw(output="text").single_string())
        print(f"Text drawing written to: {out_path}")


def main():
    test_build_draw_circuit(out_path="code.png", rounds=1)


if __name__ == "__main__":
    main()
