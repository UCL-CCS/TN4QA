from qiskit import QuantumCircuit
from qiskit.providers import Backend

class QuantumAlgorithm():

    def __init__(self, qc : QuantumCircuit) -> "QuantumAlgorithm":
        """
        Constructor for generic quantum algorithm.

        Args:
            qc: The quantum circuit for the algorithm

        Return:
            The quantum algorithm.
        """
        self.circuit = qc 
        self.num_qubits = qc.num_qubits
        self.depth = qc.depth()
        
        return 
    
    def run(self, backend : Backend, num_shots : int) -> dict[str, int]:
        """
        Execute the quantum circuit on a Qiskit backend. 
        
        Args:
            backend: The backend object.
            num_shots: The number of shots to take. 
        
        Returns:
            A dictionary of bitstring measurement results {"bs" : counts}
        """

        results = backend.run(self.circuit, shots=num_shots).result().get_counts()
        return results 
    