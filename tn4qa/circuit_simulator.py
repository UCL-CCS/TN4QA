import copy

from qiskit import QuantumCircuit

from tn4qa.mpo import MatrixProductOperator
from tn4qa.mps import MatrixProductState


class CircuitSimulator:
    """
    A class to simulate quantum circuits built using Qiskit
    """

    def __init__(
        self, circuit: QuantumCircuit, input_state: MatrixProductState | None = None
    ) -> None:
        """
        Class constructor.

        Args:
            circuit: The Qiskit QuantumCircuit object
        """
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.set_input_state(input_state)
        self.output_state = None

    def set_input_state(self, input_state: MatrixProductState | None) -> None:
        """
        Set the input state to the circuit

        Args:
            input_state: The input state, defaults to the all zero state
        """
        if not input_state:
            input_state = MatrixProductState.all_zero_mps(self.num_qubits)

        self.input_state = input_state

    def run(self, max_bond_dimension: int | None = None) -> MatrixProductState:
        """
        Execute the quantum circuit

        Args:
            max_bond_dimension: The maximum allowed bond dimension
        """
        current_state = copy.deepcopy(self.input_state)
        for inst in self.circuit.data:
            qidxs = [
                inst.qubits[i]._index + 1 for i in range(inst.operation.num_qubits)
            ]
            mpo = MatrixProductOperator.from_qiskit_gate(inst)
            current_state = current_state.apply_sub_mpo(mpo, qidxs)

        return current_state
