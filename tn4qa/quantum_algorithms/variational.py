from .qa_base import QuantumAlgorithm
from qiskit import QuantumCircuit

class VariationalAlgorithm(QuantumAlgorithm):

    def __init__(self, qc : QuantumCircuit):
        """
        Constructor for the variational algorithms class. 
        """
        super().__init__(qc)
        self.parameters = qc.parameters
        return 
    