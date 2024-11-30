from typing import TypeAlias, Union

from .qa_base import QuantumAlgorithm
from qiskit import QuantumCircuit
from qiskit.circuit import Operation, CircuitInstruction
from numpy import ndarray
from sparse import SparseArray

UnitaryOptions : TypeAlias = Union[QuantumCircuit, Operation, CircuitInstruction] # type: ignore
StateOptions : TypeAlias = Union[QuantumCircuit, ndarray, SparseArray]

class HadamardTest(QuantumAlgorithm):

    def __init__(self, unitary : UnitaryOptions, state : StateOptions) -> "HadamardTest":
        # Define number of qubits
        qc = QuantumCircuit()
        super().__init__(qc)
