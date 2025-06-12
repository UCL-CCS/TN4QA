from qiskit import QuantumCircuit

from ..base import QuantumAlgorithm


class QSCI(QuantumAlgorithm):
    def __init__(self, ansatz: QuantumCircuit) -> "QSCI":
        """
        Constructor for QSCI class.
        """
        super().__init__(ansatz)
        return
