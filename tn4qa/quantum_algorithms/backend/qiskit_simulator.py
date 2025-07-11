import copy

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .base import QuantumBackend


class QiskitSimulatorBackend(QuantumBackend):
    """
    Backend using Qiskit simulator for circuit execution
    """

    def __init__(self) -> None:
        """Constructor"""
        self._name = "qiskit_simulator"
        self._coupling_map = None
        self._basis_gates = None
        self._num_qubits = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def coupling_map(self) -> list[tuple[int, int]]:
        return self._coupling_map

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def basis_gates(self) -> list[str]:
        return self._basis_gates

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int,
    ) -> dict[str, int]:
        """Execute the circuit

        Args:
            circuit: The QuantumCircuit object to run
            shots: If provided will sample from the resulting state
            max_bond: The maximum bond dimension allowed

        Returns:
            Measurement results {bitstring : count}
        """
        sim = AerSimulator()
        qc = copy.deepcopy(circuit)
        qc.measure_all()
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()
        output = {k[::-1]: v for k, v in counts.items()}
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
        return {"backend_name": self.name}
