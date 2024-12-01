from typing import TypeAlias, Union
from .qa_base import QuantumAlgorithm
from .utils import *
from qiskit import QuantumCircuit
from qiskit.circuit import Operation, CircuitInstruction
from qiskit.circuit.library import QFT
from numpy import ndarray
from sparse import SparseArray

TypeOptions : TypeAlias = Union[QuantumCircuit, Operation, CircuitInstruction, ndarray, SparseArray] # type: ignore

class QPE(QuantumAlgorithm):

    def __init__(self, unitary : TypeOptions, state : TypeOptions, num_precision_bits : int) -> "QPE": # type: ignore
        """
        Constructor for QPE algorithm.
        """
        num_state_qubits = count_qubits(state)

        unitary_circ = to_QuantumCircuit(unitary)
        state_circ = to_QuantumCircuit(state)
        iqft = QFT(num_precision_bits, inverse=True)
        
        qc = QuantumCircuit(num_state_qubits + num_precision_bits)
        for idx in range(num_precision_bits):
            temp_qc = QuantumCircuit(num_state_qubits + num_precision_bits)
            for _ in range(2**idx):
                temp_qc.compose(unitary_circ, qubits=range(num_precision_bits, num_precision_bits+num_state_qubits), inplace=True)
            temp_qc = add_controls(temp_qc, [idx])
            qc.compose(temp_qc, inplace=True)
        qc.compose(state_circ, qubits=range(num_precision_bits, num_precision_bits+num_state_qubits), inplace=True, front=True)
        qc.append(iqft, range(num_precision_bits))

        super().__init__(qc)
        