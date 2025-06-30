from ..base import QuantumAlgorithm
from ..result import Result


class QdriftSimulation(QuantumAlgorithm):
    def __init__(self):
        pass

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
