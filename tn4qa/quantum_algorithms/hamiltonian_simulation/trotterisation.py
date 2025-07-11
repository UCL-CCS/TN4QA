from timeit import default_timer

import numpy as np
from qiskit import QuantumCircuit

from ..backend.base import QuantumBackend
from ..backend.tn_backend import TNQuantumBackend
from ..base import QuantumAlgorithm
from ..result import Result
from ..utils import calculate_exp_val, exp_pauli_string_to_circ


class TrotterSimulation(QuantumAlgorithm):
    """
    Perform Hamiltonian simulation by Trotterisation
    """

    def __init__(
        self,
        hamiltonian: dict[str, float],
        duration: float,
        error: float = 0.05,
        num_steps: int | None = None,
        backend: QuantumBackend | None = None,
    ) -> "TrotterSimulation":
        """
        Constructor for Trotter simulation class.

        Args:
            hamiltonian: The qubit Hamiltonian
            duration: The time to simulate evolution for
            num_steps: The number of Trotter steps, defaults to a sensible value
        """
        self.hamiltonian = hamiltonian

        norm = np.sum([np.abs(x) for x in hamiltonian.values()])
        self.duration = duration
        if not num_steps:
            self.num_steps = int(np.ceil((duration**2 * norm**2) / error))
        else:
            self.num_steps = num_steps
        self.delta_t = duration / self.num_steps

        pauli_strings = list(hamiltonian.keys())

        num_qubits = len(pauli_strings[0])
        qc = QuantumCircuit(num_qubits)

        for _ in range(self.num_steps):
            for p in pauli_strings:
                temp_qc = exp_pauli_string_to_circ(p, self.delta_t * hamiltonian[p])
                qc.compose(temp_qc, inplace=True)

        self._circuit = qc
        self.set_backend(backend)

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    def run(self, num_shots: int = 1024, observable: dict | None = None) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value."""
        start_time = default_timer()
        if observable is None:
            counts = self.backend.run(self.circuit, shots=num_shots)
            result = None
        else:
            result = calculate_exp_val(
                self.circuit, observable, self.backend, num_shots
            )
            counts = None

        end_time = default_timer()

        metadata = {
            "algorithm_name": "Trotterisation",
            "num_shots": num_shots,
            "total_runtime": end_time - start_time,
        }
        if self.backend is not None:
            metadata["backend_name"] = self.backend.name
            metadata["backend_coupling_map"] = self.backend.coupling_map
            metadata["backend_basis_gates"] = self.backend.basis_gates
            metadata["backend_num_qubits"] = self.backend.num_qubits
        result = Result(
            result=result,
            measurements=counts,
            parameters=None,
            metadata=metadata,
        )
        return result

    def set_backend(self, backend: QuantumBackend | None = None) -> None:
        """Attach a QuantumBackend instance for execution."""
        if backend is None:
            backend = TNQuantumBackend()
        self.backend = backend
        return
