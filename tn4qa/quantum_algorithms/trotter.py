from .qa_base import QuantumAlgorithm
from .utils import *
from qiskit import QuantumCircuit

class TrotterSimulation(QuantumAlgorithm):

    def __init__(self, hamiltonian : dict[str, complex], duration : float, num_steps : int) -> "TrotterSimulation":
        """
        Constructor for Trotter simulation class.
        """
        pauli_strings = list(hamiltonian.keys())

        num_qubits = len(pauli_strings[0])
        qc = QuantumCircuit(num_qubits)

        timestep = duration / num_steps 
        for _ in range(num_steps):
            for p in pauli_strings:
                temp_qc = pauli_string_to_circ(p, timestep*hamiltonian[p])
                qc.compose(temp_qc, inplace=True)

        super().__init__(qc)
        return