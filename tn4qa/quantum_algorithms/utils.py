import copy
import inspect
from typing import Callable, List, TypeAlias, Union

import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Operation
from qiskit.circuit.library import UnitaryGate
from sparse import SparseArray

from .backend.base import QuantumBackend

QiskitOptions: TypeAlias = Union[QuantumCircuit, Operation, CircuitInstruction]  # type: ignore
ArrayOptions = TypeAlias = Union[ndarray, SparseArray]


def count_qubits(obj: QiskitOptions | ArrayOptions) -> int:  # type: ignore
    """
    Count the number of qubits from an object.

    Args:
        obj: The circuit or array to count the qubits of

    Returns:
        The number of qubits acted on by obj
    """
    if isinstance(obj, QiskitOptions):
        num_qubits = obj.num_qubits
    elif isinstance(obj, ArrayOptions):
        num_qubits = int(np.log2(obj.shape[0]))

    return num_qubits


def to_quantum_circuit(obj: QiskitOptions | ArrayOptions) -> QuantumCircuit:  # type: ignore
    """
    Convert an object to a QuantumCircuit.

    Args:
        obj: The instruction or array to convert to a QuantumCircuit

    Returns:
        A QuantumCircuit representation of obj
    """
    num_qubits = count_qubits(obj)

    if isinstance(obj, QuantumCircuit):
        return obj
    elif isinstance(obj, Operation | CircuitInstruction):
        qc = QuantumCircuit(num_qubits)
        qc.append(obj, range(num_qubits))
        return qc
    elif isinstance(obj, ndarray):
        qc = QuantumCircuit(num_qubits)
        qc.append(UnitaryGate(obj), list(range(num_qubits))[::-1])
        return qc
    elif isinstance(obj, SparseArray):
        qc = QuantumCircuit(num_qubits)
        qc.append(UnitaryGate(obj.todense()), list(range(num_qubits))[::-1])
        return qc


def add_controls(qc: QuantumCircuit, ctrl_idxs: List[int]) -> QuantumCircuit:
    """
    Replace every instruction in qc with a controlled instruction on ctrl_idx.

    Args:
        qc: A quantum circuit
        ctrl_idxs: The indices of qubits to act as control bits

    Returns:
        A new QuantumCircuit
    """
    ctrl_qc = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        qubits = [inst.qubits[i]._index for i in range(len(inst.qubits))][
            ::-1
        ] + ctrl_idxs
        ctrl_inst = inst.operation.control(len(ctrl_idxs))
        ctrl_qc.append(ctrl_inst, qubits[::-1])
    return ctrl_qc


def exp_pauli_string_to_circ(pauli_string: str, rot_angle: float) -> QuantumCircuit:
    """
    Create a circuit for an exponential Pauli string.

    Args:
        pauli_string: The Pauli string
        rot_angle: The rotation angle in the exponential

    Returns:
        A QuantumCircuit
    """
    qc = QuantumCircuit(len(pauli_string))

    if pauli_string == "I" * len(pauli_string):
        return qc

    for p_idx in range(len(pauli_string)):
        p = pauli_string[p_idx]
        if p == "X":
            qc.h(p_idx)
        elif p == "Y":
            qc.sdg(p_idx)
            qc.h(p_idx)

    non_id_qubits = [
        p_idx for p_idx in range(len(pauli_string)) if pauli_string[p_idx] != "I"
    ]
    for non_id_idx in range(len(non_id_qubits) - 1):
        q1, q2 = non_id_qubits[non_id_idx], non_id_qubits[non_id_idx + 1]
        qc.cx(q1, q2)
    qc.rz(2 * rot_angle, non_id_qubits[-1])
    for non_id_idx in range(len(non_id_qubits) - 1):
        q1, q2 = non_id_qubits[non_id_idx], non_id_qubits[non_id_idx + 1]
        qc.cx(q1, q2)

    for p_idx in range(len(pauli_string)):
        p = pauli_string[p_idx]
        if p == "X":
            qc.h(p_idx)
        elif p == "Y":
            qc.h(p_idx)
            qc.s(p_idx)

    return qc


def controlled_exp_pauli_string_circ(
    pauli_string: str, rot_angle: float, ctrl_idxs: list[int]
) -> QuantumCircuit:
    """
    Create a controlled circuit for an exponential Pauli string.

    Args:
        pauli_string: The Pauli string
        rot_angle: The rotation angle in the exponential
        ctrl_idxs: The list of qubits to act as controls

    Returns:
        A QuantumCircuit
    """
    qc = QuantumCircuit(len(pauli_string))

    for p_idx in range(len(pauli_string)):
        p = pauli_string[p_idx]
        if p == "X":
            qc.h(p_idx)
        elif p == "Y":
            qc.sdg(p_idx)
            qc.h(p_idx)

    non_id_qubits = [
        p_idx for p_idx in range(len(pauli_string)) if pauli_string[p_idx] != "I"
    ]
    for non_id_idx in range(len(non_id_qubits) - 1):
        q1, q2 = non_id_qubits[non_id_idx], non_id_qubits[non_id_idx + 1]
        qc.cx(q1, q2)
    qc.crz(2 * rot_angle, ctrl_idxs, non_id_idx[-1])
    for non_id_idx in range(len(non_id_qubits) - 1):
        q1, q2 = non_id_qubits[non_id_idx], non_id_qubits[non_id_idx + 1]
        qc.cx(q1, q2)

    for p_idx in range(len(pauli_string)):
        p = pauli_string[p_idx]
        if p == "X":
            qc.h(p_idx)
        elif p == "Y":
            qc.h(p_idx)
            qc.s(p_idx)

    return qc


def calculate_exp_val(
    circuit: QuantumCircuit, observable: dict, backend: QuantumBackend, shots: int
) -> float:
    """Calculate an expectation value from a circuit

    Args:
        circuit: The QuantumCircuit
        observable: The observable (sum of Pauli terms)
        backend: The backend
        shots: Shots per circuit
    """

    def filter_counts(counts: dict, relevant_qidxs: list[int]):
        new_counts = {}
        for k, v in counts.items():
            new_k = ""
            for b_idx in range(len(k)):
                b = k[b_idx]
                if b_idx in relevant_qidxs:
                    new_k += b
            if new_k in new_counts:
                new_counts[new_k] += v
            else:
                new_counts[new_k] = v
        return new_counts

    def parity_of_bitstring(bitstring: str):
        parity = 1
        for b in bitstring:
            if b == "1":
                parity *= -1
        return parity

    exp_val = 0
    for k, v in observable.items():
        if k == "I" * len(k):
            exp_val += v
            continue

        qc = copy.deepcopy(circuit)
        relevant_qidxs = []
        for p_idx in range(len(k)):
            p = k[p_idx]
            if p == "X":
                qc.h(p_idx)
                relevant_qidxs.append(p_idx)
            elif p == "Y":
                qc.sdg(p_idx)
                qc.h(p_idx)
                relevant_qidxs.append(p_idx)
            elif p == "Z":
                relevant_qidxs.append(p_idx)
        counts = backend.run(qc, shots=shots)
        filtered_counts = filter_counts(counts, relevant_qidxs)
        counts_p = 0
        counts_n = 0
        for a, b in filtered_counts.items():
            parity = parity_of_bitstring(a)
            if parity == 1:
                counts_p += b
            else:
                counts_n += b
        exp_val += v * ((counts_p - counts_n) / shots)

    return exp_val


def calculate_exp_val_stochastic(
    circuit_builder: Callable,
    observable: dict,
    backend: QuantumBackend,
    shots: int,
    *args,
    **kwargs,
) -> float:
    """Calculate an expectation value from a circuit

    Args:
        circuit_builder: The QuantumCircuit builder function
        observable: The observable (sum of Pauli terms)
        backend: The backend
        shots: Shots per circuit
        args: For circuit builder function
        kwargs: For circuit builder function
    """

    def filter_counts(counts: dict, relevant_qidxs: list[int]):
        new_counts = {}
        for k, v in counts.items():
            new_k = ""
            for b_idx in range(len(k)):
                b = k[b_idx]
                if b_idx in relevant_qidxs:
                    new_k += b
            if new_k in new_counts:
                new_counts[new_k] += v
            else:
                new_counts[new_k] = v
        return new_counts

    def parity_of_bitstring(bitstring: str):
        parity = 1
        for b in bitstring:
            if b == "1":
                parity *= -1
        return parity

    exp_val = 0
    for k, v in observable.items():
        if k == "I" * len(k):
            exp_val += v
            continue

        counts = {}
        for _ in range(shots):
            sig = inspect.signature(circuit_builder)
            if len(sig.parameters) == 0:
                qc = circuit_builder()
            else:
                qc = circuit_builder(*args, **kwargs)
            relevant_qidxs = []
            for p_idx in range(len(k)):
                p = k[p_idx]
                if p == "X":
                    qc.h(p_idx)
                    relevant_qidxs.append(p_idx)
                elif p == "Y":
                    qc.sdg(p_idx)
                    qc.h(p_idx)
                    relevant_qidxs.append(p_idx)
                elif p == "Z":
                    relevant_qidxs.append(p_idx)
            bs = list(backend.run(qc, shots=1).keys())[0]
            if bs in counts:
                counts[bs] += 1
            else:
                counts[bs] = 1

        filtered_counts = filter_counts(counts, relevant_qidxs)
        counts_p = 0
        counts_n = 0
        for a, b in filtered_counts.items():
            parity = parity_of_bitstring(a)
            if parity == 1:
                counts_p += b
            else:
                counts_n += b
        exp_val += v * ((counts_p - counts_n) / shots)

    return exp_val
