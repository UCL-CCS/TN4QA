from typing import TypeAlias, Union

from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Operation
from qiskit.circuit.library import QFT
from sparse import SparseArray

from ..base import QuantumAlgorithm
from ..result import Result
from ..utils import add_controls, count_qubits, to_QuantumCircuit

TypeOptions: TypeAlias = Union[
    QuantumCircuit, Operation, CircuitInstruction, ndarray, SparseArray  # type: ignore
]  # type: ignore


class QPE(QuantumAlgorithm):
    def __init__(
        self, unitary: TypeOptions, state: TypeOptions, num_precision_bits: int
    ) -> "QPE":  # type: ignore
        """
        Constructor for QPE algorithm.

        Args:
            unitary: The unitary operation to estimate the phases for
            state: The input state to QPE
            num_precision_bits: The number of precision bits
        """
        num_state_qubits = count_qubits(state)

        unitary_circ = to_QuantumCircuit(unitary)
        state_circ = to_QuantumCircuit(state)
        iqft = QFT(num_precision_bits, inverse=True)

        qc = QuantumCircuit(num_state_qubits + num_precision_bits)
        for idx in range(num_precision_bits):
            temp_qc = QuantumCircuit(num_state_qubits + num_precision_bits)
            for _ in range(2**idx):
                temp_qc.compose(
                    unitary_circ,
                    qubits=range(
                        num_precision_bits, num_precision_bits + num_state_qubits
                    ),
                    inplace=True,
                )
            temp_qc = add_controls(temp_qc, [idx])
            qc.compose(temp_qc, inplace=True)
        qc.compose(
            state_circ,
            qubits=range(num_precision_bits, num_precision_bits + num_state_qubits),
            inplace=True,
            front=True,
        )
        qc.append(iqft, range(num_precision_bits))

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
