from qiskit import QuantumCircuit

from ..mps import MatrixProductOperator
from ..tn import TensorNetwork


class MPOOptimiser:
    """
    A class for locally optimising a quantum circuit with respect to a reference MPO and the HS distance
    """

    def __init__(self, qc: QuantumCircuit, reference: MatrixProductOperator) -> None:
        """
        Constructor

        Args:
            qc: The quantum circuit that will be optimised
            reference: The reference MPO
        """
        self.tn = TensorNetwork.from_qiskit_circuit(qc)
        self.tn_dag = TensorNetwork.from_qiskit_circuit(qc, dagger=True)
        for t in self.tn.tensors:
            t.labels.append(f"variational_site_{self.tn.tensors.index(t)+1}")
        for t in self.tn_dag.tensors:
            t.labels.append(
                f"variational_site_{len(self.tn_dag.tensors)-self.tn_dag.tensors.index(t)}"
            )
        self.reference = reference
