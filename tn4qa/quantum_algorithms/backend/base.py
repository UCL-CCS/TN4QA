from abc import ABC, abstractmethod


class QuantumBackend(ABC):
    """
    A class for quantum backends (simulated and real)
    """

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
