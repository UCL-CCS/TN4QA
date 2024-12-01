from .variational import VariationalAlgorithm
from qiskit import QuantumCircuit

class VQE(VariationalAlgorithm):

    def __init__(self, ansatz : QuantumCircuit) -> "VQE":
        super().__init__(ansatz)
        return 