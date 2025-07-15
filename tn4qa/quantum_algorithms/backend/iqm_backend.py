import copy

from iqm.qiskit_iqm import IQMProvider
from qiskit import QuantumCircuit, transpile

from .base import QuantumBackend


class IQMBackend(QuantumBackend):
    """
    Backend using IQM device for circuit execution
    """

    def __init__(self, token: str, device: str) -> None:
        """Constructor

        Args:
            token: Access token
            device: Name of device, either emerald or sirius"""
        self._name = "IQM_" + device
        self.url = "https://cocos.resonance.meetiqm.com/" + device
        self.provider = IQMProvider(self.url, token=token)
        self.backend = self.provider.get_backend()
        self._coupling_map = self.backend.coupling_map
        self._basis_gates = self.backend.operation_names
        self._num_qubits = self.backend.num_qubits

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
        qc = copy.deepcopy(circuit)
        qc.measure_all()
        qc = transpile(qc, backend=self.backend)
        result = self.backend.run(qc, shots=shots).result()
        counts = result.get_counts()
        counts = {k[::-1]: v for k, v in counts.items()}
        return counts

    def parse_openqasm(self, filename: str) -> QuantumCircuit:
        """Parse an OpenQASM input circuit

        Args:
            filename: The filename of the OpenQASM input

        Returns:
            A qiskit QuantumCircuit object
        """
        qc = QuantumCircuit.from_qasm_file(filename)
        qc = transpile(qc, backend=self.backend)
        return qc

    def get_device_info(self) -> dict:
        """Return a dictionary describing the backend."""
        return {"backend_name": self.name}
