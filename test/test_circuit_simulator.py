import numpy as np
from qiskit import QuantumCircuit

from tn4qa.circuit_simulator import CircuitSimulator


def test_basic_circuit():
    qc = QuantumCircuit(5)
    qc.h(0)
    for idx in range(4):
        qc.cx(idx, idx + 1)

    sim = CircuitSimulator(qc)
    output = sim.run()

    expected_output = np.array([np.sqrt(1 / 2)] + [0.0] * 30 + [np.sqrt(1 / 2)])

    assert np.allclose(output.to_dense_array(), expected_output)


def test_reverse_circuit():
    qc = QuantumCircuit(5)
    qc.h(4)
    for idx in [4, 3, 2, 1]:
        qc.cx(idx, idx - 1)

    sim = CircuitSimulator(qc)
    output = sim.run()

    expected_output = np.array([np.sqrt(1 / 2)] + [0.0] * 30 + [np.sqrt(1 / 2)])

    assert np.allclose(output.to_dense_array(), expected_output)


def test_nonlinear_circuit():
    qc = QuantumCircuit(6)

    # Apply circuit
    for idx in range(6):
        qc.h(idx)
    qc.cx(0, 2)
    qc.cx(4, 1)
    qc.cz(1, 3)
    qc.cz(2, 3)
    for idx in range(4):
        qc.x(idx)
    for idx in range(4):
        qc.y(idx + 2)
    qc.cx(5, 3)
    qc.cx(1, 3)
    qc.cx(3, 4)
    qc.cz(0, 5)

    # Undo circuit
    qc.cz(0, 5)
    qc.cx(3, 4)
    qc.cx(1, 3)
    qc.cx(5, 3)
    for idx in range(4):
        qc.y(idx + 2)
    for idx in range(4):
        qc.x(idx)
    qc.cz(2, 3)
    qc.cz(1, 3)
    qc.cx(4, 1)
    qc.cx(0, 2)
    for idx in range(6):
        qc.h(idx)

    # Sim
    sim = CircuitSimulator(qc)
    output = sim.run()

    expected_output = np.array([1.0] + [0.0] * 63)

    assert np.allclose(output.to_dense_array(), expected_output)
