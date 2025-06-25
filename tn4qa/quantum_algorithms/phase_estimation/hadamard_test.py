from typing import TypeAlias, Union

from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Operation
from sparse import SparseArray

from ..base import QuantumAlgorithm
from ..result import Result
from ..utils import add_controls, count_qubits, to_QuantumCircuit

TypeOptions: TypeAlias = Union[
    QuantumCircuit, Operation, CircuitInstruction, ndarray, SparseArray
]  # type: ignore


class HadamardTest(QuantumAlgorithm):
    def __init__(self, unitary: TypeOptions, state: TypeOptions) -> "HadamardTest":  # type: ignore
        num_state_qubits = count_qubits(state)

        state_circ = to_QuantumCircuit(state)
        unitary_circ = to_QuantumCircuit(unitary)

        qc = QuantumCircuit(num_state_qubits + 1)
        qc.compose(unitary_circ, qubits=range(1, num_state_qubits + 1), inplace=True)
        qc = add_controls(qc, [0])
        qc.compose(
            state_circ, qubits=range(1, num_state_qubits + 1), inplace=True, front=True
        )

        super().__init__(qc)

    def run(self, **kwargs) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value."""
        pass

    def construct_circuit(self, **kwargs):
        """Return the circuit(s) that represent the quantum part of the algorithm."""
        pass

    def set_backend(self, backend, **kwargs) -> None:
        """Attach a QuantumBackend instance for execution."""
        pass

    def get_result(self):
        """Return structured results."""
        pass
