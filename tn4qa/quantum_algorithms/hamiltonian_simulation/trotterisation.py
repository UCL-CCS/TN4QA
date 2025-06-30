from qiskit import QuantumCircuit

from ..base import QuantumAlgorithm
from ..result import Result
from ..utils import exp_pauli_string_to_circ


class TrotterSimulation(QuantumAlgorithm):
    """
    Perform Hamiltonian simulation by Trotterisation
    """

    def __init__(
        self, hamiltonian: dict[str, complex], duration: float, num_steps: int
    ) -> "TrotterSimulation":
        """
        Constructor for Trotter simulation class.

        Args:
            hamiltonian: The qubit Hamiltonian
            duration: The time to simulate evolution for
            num_steps: The number of Trotter steps
        """
        pauli_strings = list(hamiltonian.keys())

        num_qubits = len(pauli_strings[0])
        qc = QuantumCircuit(num_qubits)

        timestep = duration / num_steps
        for _ in range(num_steps):
            for p in pauli_strings:
                temp_qc = exp_pauli_string_to_circ(p, timestep * hamiltonian[p])
                qc.compose(temp_qc, inplace=True)

        super().__init__(qc)

    def run(self, **kwargs) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value."""
        pass

    def construct_circuit(self, **kwargs):
        """Return the circuit(s) that represent the quantum part of the algorithm."""
        pass

    def set_backend(self, backend, **kwargs) -> None:
        """Attach a QuantumBackend instance for execution."""
        pass

    def get_result(self):
        """Return structured results."""
        pass
