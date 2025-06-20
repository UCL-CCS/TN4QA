import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    CXGate,
    CZGate,
    ExcitationPreserving,
    PauliTwoDesign,
    RXGate,
    RYGate,
    RZGate,
)
from qiskit.quantum_info import Operator, random_unitary


def hea_ansatz(
    n_qubits: int, layers: int, sq_rotations: list[str], mq_gate: str
) -> QuantumCircuit:
    """
    Hardware Efficient Ansatz.

    Args:
        n_qubits: The number of qubits
        layers: The number of layers in the ansatz
        sq_rotations: The list of single-qubit operations available on the hardware
        mq_gate: The multi-qubit (generally 2q) gate available on the hardware

    Returns:
        QuantumCircuit
    """
    hea = QuantumCircuit(n_qubits)
    num_params = 3 * n_qubits * layers
    params = [Parameter(str(a)) for a in range(num_params)]

    def get_sq_gate(idx, param):
        sq = sq_rotations[idx]
        if "rx" == sq:
            return RXGate(param)
        if "ry" == sq:
            return RYGate(param)
        if "rz" == sq:
            return RZGate(param)

    if mq_gate == "cx":
        mq_gate = CXGate()
    elif mq_gate == "cz":
        mq_gate = CZGate()

    for _ in range(layers):
        for q_idx in range(n_qubits):
            gate = get_sq_gate(0, params[0])
            hea.append(gate, q_idx)
            params.pop(0)
        for q_idx in range(n_qubits):
            gate = get_sq_gate(1, params[0])
            hea.append(gate, q_idx)
            params.pop(0)
        for q_idx in range(n_qubits):
            gate = get_sq_gate(0, params[0])
            hea.append(gate, q_idx)
            params.pop(0)

        for q_idx in range(n_qubits - 1):
            hea.append(mq_gate, (q_idx, q_idx + 1))
        hea.append(mq_gate, (n_qubits - 1, 0))

    return hea


def pauli_two_design_ansatz(n_qubits: int, layers: int) -> QuantumCircuit:
    """
    Pauli two design ansatz.

    Args:
        n_qubits: The number of qubits
        layers: The number of layers in the ansatz

    Returns:
        QuantumCircuit
    """
    return PauliTwoDesign(n_qubits, reps=layers).decompose()


def number_preserving_ansatz(
    n_qubits: int, layers: int, entanglement: str
) -> QuantumCircuit:
    """
    Number preserving ansatz.

    Args:
        n_qubits: The number of qubits
        layers: The number of layers in the ansatz
        entanglement: What entanglement structure to use

    Returns:
        QuantumCircuit
    """
    return ExcitationPreserving(
        n_qubits, reps=layers, entanglement=entanglement
    ).decompose()


def identity_brickwork_circuit_offset(n_qubits: int, layers: int) -> QuantumCircuit:
    """
    Brickwork circuit coupling alpha-alpha electrons and beta-beta electrons

    Args:
        n_qubits: number of qubits
        layers: number of layers

    Returns:
        QuantumCircuit
    """
    id_gate = Operator(np.eye(4)).to_instruction()
    qc = QuantumCircuit(n_qubits)
    for _ in range(layers):
        for idx in range(int(n_qubits / 2)):
            qidxs = [2 * idx + x for x in [0, 2]]
            qc.append(id_gate, qidxs)
        for idx in range(int(n_qubits / 2)):
            qidxs = [2 * idx + x for x in [1, 3]]
            qc.append(id_gate, qidxs)
    return qc


def random_brickwork_circuit_offset(n_qubits: int, layers: int) -> QuantumCircuit:
    """
    Brickwork circuit of random gates coupling alpha-alpha electrons and beta-beta electrons

    Args:
        n_qubits: The number of qubits
        layers: The number of layers

    Returns:
        A quantum circuit
    """
    qc = QuantumCircuit(n_qubits)
    for _ in range(layers):
        for idx in range(int(n_qubits / 4)):
            random_gate = random_unitary(4)
            qidxs = [2 * idx + x for x in [0, 2]]
            qc.append(random_gate, qidxs)
        for idx in range(int(n_qubits / 4)):
            random_gate = random_unitary(4)
            qidxs = [2 * idx + x for x in [1, 3]]
            qc.append(random_gate, qidxs)
    return qc


def identity_brickwork_circuit(
    n_qubits: int, layers: int, qubits_per_gate: int = 2, gap: int = 1
) -> QuantumCircuit:
    """
    Brickwork circuit of identity gates.

    Args:
        n_qubits: The number of qubits
        layers: The number of layers
        qubits_per_gate: The number of qubits per gate
        gap: The number of qubits to skip between layers

    Returns:
        A quantum circuit
    """
    id_gate = Operator(np.eye(2**qubits_per_gate)).to_instruction()
    qc = QuantumCircuit(n_qubits)
    for _ in range(layers):
        for idx in range(int(np.floor(n_qubits / qubits_per_gate))):
            qidxs = [qubits_per_gate * idx + x for x in range(qubits_per_gate)]
            qc.append(id_gate, qidxs)
        for idx in range(int(np.floor((n_qubits - gap) / qubits_per_gate))):
            qidxs = [qubits_per_gate * idx + gap + x for x in range(qubits_per_gate)]
            qc.append(id_gate, qidxs)
    return qc


def random_brickwork_circuit(
    n_qubits: int, layers: int, qubits_per_gate: int = 2, gap: int = 1
) -> QuantumCircuit:
    """
    Brickwork circuit of random gates.

    Args:
        n_qubits: The number of qubits
        layers: The number of layers
        qubits_per_gate: The number of qubits per gate
        gap: The number of qubits to skip between layers

    Returns:
        A quantum circuit
    """
    qc = QuantumCircuit(n_qubits)
    for _ in range(layers):
        for idx in range(int(np.floor(n_qubits / qubits_per_gate))):
            random_gate = random_unitary(2**qubits_per_gate)
            qidxs = [qubits_per_gate * idx + x for x in range(qubits_per_gate)]
            qc.append(random_gate, qidxs)
        for idx in range(int(np.floor((n_qubits - gap) / qubits_per_gate))):
            random_gate = random_unitary(2**qubits_per_gate)
            qidxs = [qubits_per_gate * idx + gap + x for x in range(qubits_per_gate)]
            qc.append(random_gate, qidxs)
    return qc


def identity_staircase_circuit(
    n_qubits: int, layers: int, qubits_per_gate: int
) -> QuantumCircuit:
    """
    Staircase circuit of identity gates.

    Args:
        n_qubits: The number of qubits
        layers: The number of layers
        qubits_per_gate: The number of qubits per gate

    Returns:
        A quantum circuit
    """
    id_gate = Operator(np.eye(2**qubits_per_gate)).to_instruction()
    qc = QuantumCircuit(n_qubits)
    for _ in range(layers):
        for idx in range(n_qubits + 1 - qubits_per_gate):
            qidxs = [idx + x for x in range(qubits_per_gate)]
            qc.append(id_gate, qidxs)
    return qc


def random_staircase_circuit(
    n_qubits: int, layers: int, qubits_per_gate: int
) -> QuantumCircuit:
    """
    Staircase circuit of random gates.

    Args:
        n_qubits: The number of qubits
        layers: The number of layers
        qubits_per_gate: The number of qubits per gate

    Returns:
        A quantum circuit
    """
    qc = QuantumCircuit(n_qubits)
    for _ in range(layers):
        for idx in range(n_qubits + 1 - qubits_per_gate):
            random_gate = random_unitary(2**qubits_per_gate)
            qidxs = [idx + x for x in range(qubits_per_gate)]
            qc.append(random_gate, qidxs)
    return qc
