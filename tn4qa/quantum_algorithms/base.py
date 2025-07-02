from abc import ABC, abstractmethod

from .result import Result


class QuantumAlgorithm(ABC):
    """
    Abstract base class for quantum algorithms.
    """

    @abstractmethod
    def run(self, **kwargs) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value."""
        pass

    # @abstractmethod
    # def construct_circuit(self, **kwargs):
    #     """Return the circuit(s) that represent the quantum part of the algorithm."""
    #     pass

    @abstractmethod
    def set_backend(self, backend, **kwargs) -> None:
        """Attach a QuantumBackend instance for execution."""
        pass

    # @abstractmethod
    # def get_result(self):
    #     """Return structured results."""
    #     pass
