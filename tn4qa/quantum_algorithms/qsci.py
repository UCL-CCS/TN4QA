from .variational import VariationalAlgorithm
from qiskit import QuantumCircuit

class QSCI(VariationalAlgorithm):

    def __init__(self, ansatz : QuantumCircuit) -> "QSCI":
        """
        Constructor for QSCI class.
        """
        super().__init__(ansatz)
        return