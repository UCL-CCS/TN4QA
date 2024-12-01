from qiskit import QuantumCircuit
from qiskit.circuit import Operation, CircuitInstruction
from qiskit.circuit.library import UnitaryGate
from numpy import ndarray 
from sparse import SparseArray
import numpy as np 
from typing import TypeAlias, Union, List

QiskitOptions : TypeAlias = Union[QuantumCircuit, Operation, CircuitInstruction] # type: ignore
ArrayOptions = TypeAlias = Union[ndarray, SparseArray]

def count_qubits(obj : QiskitOptions | ArrayOptions) -> int: # type: ignore
    """
    Count the number of qubits from an object.
    """
    if isinstance(obj, QiskitOptions):
        num_qubits = obj.num_qubits
    elif isinstance(obj, ArrayOptions):
        num_qubits = int(np.log2(obj.shape[0]))

    return num_qubits

def to_QuantumCircuit(obj : QiskitOptions | ArrayOptions) -> QuantumCircuit: # type: ignore
    """
    Convert an object to a QuantumCircuit.
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
        qc.append(UnitaryGate(obj), range(num_qubits))
        return qc
    elif isinstance(obj, SparseArray):
        qc = QuantumCircuit(num_qubits)
        qc.append(UnitaryGate(obj.todense()), range(num_qubits))
        return qc 
    
def add_controls(qc : QuantumCircuit, ctrl_idxs : List[int]) -> QuantumCircuit:
    """
    Replace every instruction in qc with a controlled instruction on ctrl_idx.
    """
    ctrl_qc = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        qubits = [inst.qubits[i]._index for i in range(len(inst.qubits))][::-1] + ctrl_idxs
        ctrl_inst = inst.operation.control(len(ctrl_idxs))
        ctrl_qc.append(ctrl_inst, qubits[::-1])
    return ctrl_qc

def pauli_string_to_circ(pauli_string : str, rot_angle : float) -> QuantumCircuit:
    """
    Create a circuit for an exponential Pauli string.
    """
    qc = QuantumCircuit(len(pauli_string))

    for p_idx in range(len(pauli_string)):
        p = pauli_string[p_idx]
        if p == "X":
            qc.h(p_idx)
        elif p == "Y":
            qc.sdg(p_idx)
            qc.h(p_idx)

    non_id_qubits = [p_idx for p_idx in range(len(pauli_string)) if pauli_string[p_idx] != "I"]
    for non_id_idx in range(len(non_id_qubits)):
        q1, q2 = non_id_qubits[non_id_idx], non_id_qubits[non_id_idx+1]
        qc.cx(q1, q2)
    qc.rz(2*rot_angle)
    for non_id_idx in range(len(non_id_qubits)):
        q1, q2 = non_id_qubits[non_id_idx], non_id_qubits[non_id_idx+1]
        qc.cx(q1, q2)
        
    for p_idx in range(len(pauli_string)):
        p = pauli_string[p_idx]
        if p == "X":
            qc.h(p_idx)
        elif p == "Y":
            qc.h(p_idx)
            qc.s(p_idx)
