from qiskit import QuantumCircuit

from ...circuit_simulator import CircuitSimulator
from ...mps import MatrixProductState
from .base import QuantumBackend


class TNQuantumBackend(QuantumBackend):
    """
    Backend using TN4QA's CircuitSimulator for circuit execution
    """

    def __init__(self) -> None:
        """Constructor"""
        self.backend_name = "tn4qa_circuit_simulator"

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int,
        max_bond: int | None = None,
        input_state: MatrixProductState | None = None,
    ) -> dict[str, int]:
        """Execute the circuit

        Args:
            circuit: The QuantumCircuit object to run
            shots: If provided will sample from the resulting state
            max_bond: The maximum bond dimension allowed

        Returns:
            Measurement results {bitstring : count}
        """
        sim = CircuitSimulator(circuit, input_state=input_state)
        output = sim.run(max_bond_dimension=max_bond, samples=shots)
        return output

    def parse_openqasm(self, filename: str) -> QuantumCircuit:
        """Parse an OpenQASM input circuit

        Args:
            filename: The filename of the OpenQASM input

        Returns:
            A qiskit QuantumCircuit object
        """
        qc = QuantumCircuit.from_qasm_file(filename)
        return qc

    def get_device_info(self) -> dict:
        """Return a dictionary describing the backend."""
        return {"backend_name": self.backend_name}
