from typing import TypeAlias, Union
from .qa_base import QuantumAlgorithm
from .utils import *
from qiskit import QuantumCircuit
from qiskit.circuit import Operation, CircuitInstruction
from numpy import ndarray
from sparse import SparseArray

TypeOptions : TypeAlias = Union[QuantumCircuit, Operation, CircuitInstruction, ndarray, SparseArray] # type: ignore

class HadamardTest(QuantumAlgorithm):

    def __init__(self, unitary : TypeOptions, state : TypeOptions) -> "HadamardTest": # type: ignore
        
        num_state_qubits = count_qubits(state)

        state_circ = to_QuantumCircuit(state)
        unitary_circ = to_QuantumCircuit(unitary)

        qc = QuantumCircuit(num_state_qubits + 1)
        qc.compose(unitary_circ, qubits=range(1, num_state_qubits+1), inplace=True)
        qc = add_controls(qc, [0])
        qc.compose(state_circ, qubits=range(1, num_state_qubits+1), inplace=True, front=True)

        super().__init__(qc)
