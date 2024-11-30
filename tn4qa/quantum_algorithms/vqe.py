from .variational import VariationalAlgorithm
from qiskit import QuantumCircuit

class VQE(VariationalAlgorithm):

    def __init__(self, qc : QuantumCircuit) -> "VQE":
        super().__init__(qc)
        return 