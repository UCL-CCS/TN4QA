import copy

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from tn4qa.mpo import MatrixProductOperator
from tn4qa.mps import MatrixProductState
from tn4qa.tensor import Tensor
from tn4qa.tn import TensorNetwork


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
            indices = [f"O{qidxs[i]}" for i in range(inst.operation.num_qubits)] + [
                f"I{qidxs[i]}" for i in range(inst.operation.num_qubits)
            ]
            if len(qidxs) == 2:
                tensor = Tensor.from_qiskit_gate(inst, indices=indices)
                tn = TensorNetwork([tensor])
                tn.svd(
                    tn.tensors[0],
                    input_indices=[indices[0], indices[2]],
                    output_indices=[indices[1], indices[3]],
                    new_index_name=f"C{qidxs[0]}",
                )
                tn.tensors[0].reorder_indices(
                    [f"C{qidxs[0]}", f"O{qidxs[0]}", f"I{qidxs[0]}"]
                )
                tn.tensors[1].reorder_indices(
                    [f"C{qidxs[0]}", f"O{qidxs[1]}", f"I{qidxs[1]}"]
                )
                arrays = [tn.tensors[i].data for i in range(2)]
            else:
                arrays = [Operator(inst.operation).reverse_qargs().data]
            mpo = MatrixProductOperator.from_arrays(arrays)
            current_state = current_state.apply_sub_mpo(mpo, qidxs)

        return current_state
