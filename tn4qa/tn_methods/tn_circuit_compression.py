from qiskit import QuantumCircuit


class TNCircuitCompression:
    """
    A class for compressing quantum circuits by optimising a shallow TN circuit
    """

    def __init__(self, qc: QuantumCircuit) -> None:
        """
        Constructor

        Args:
            qc: The QuantumCircuit to compress
        """
