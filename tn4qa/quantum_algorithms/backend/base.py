from abc import ABC, abstractmethod


class QuantumBackend(ABC):
    """
    A class for quantum backends (simulated and real)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name"""
        pass

    @property
    @abstractmethod
    def coupling_map(self) -> str:
        """Return the backend coupling map"""
        pass

    @property
    @abstractmethod
    def num_qubits(self) -> str:
        """Return the backend num qubits"""
        pass

    @property
    @abstractmethod
    def basis_gates(self) -> str:
        """Return the backend basis gates"""
        pass

    @abstractmethod
    def run(self, circuit, shots: int, **kwargs) -> dict[str, int]:
        """Execute the circuit"""
        pass

    @abstractmethod
    def parse_openqasm(self, filename: str):
        """Parse an OpenQASM input circuit"""
        pass

    @abstractmethod
    def get_device_info(self) -> dict:
        """Return a dictionary describing the backend device (connectivity, noise, etc)."""
        pass
