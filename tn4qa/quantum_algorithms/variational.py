from .qa_base import QuantumAlgorithm
from qiskit import QuantumCircuit

class VariationalAlgorithm(QuantumAlgorithm):

    def __init__(self, ansatz : QuantumCircuit):
        """
        Constructor for the variational algorithms class. 
        """
        super().__init__(ansatz)
        self.parameters = ansatz.parameters
        return 
    