from timeit import default_timer

import numpy as np
from qiskit import QuantumCircuit

from ..backend.base import QuantumBackend
from ..backend.tn_backend import TNQuantumBackend
from ..base import QuantumAlgorithm
from ..result import Result
from ..utils import calculate_exp_val, exp_pauli_string_to_circ


class QDriftSimulation(QuantumAlgorithm):
    """
    Perform Hamiltonian simulation using qDRIFT
    """

    def __init__(
        self,
        hamiltonian: dict[str, complex],
        duration: float,
        error: float | None = None,
        backend: QuantumBackend | None = None,
    ) -> "QDriftSimulation":
        """
        Constructor for qDRIFT simulation class.

        Args:
            hamiltonian: The qubit Hamiltonian
            duration: The time to simulate evolution for
            num_steps: The number of Trotter steps, defaults to a sensible value
            error: The desired error
        """
        self.hamiltonian = hamiltonian
        self.norm = np.sum([np.abs(x) for x in hamiltonian.values()])
        self.duration = duration

        if error is None:
            self.error = 1e-3
        else:
            self.error = error
        self.num_terms = int(
            np.ceil(2 * (self.norm**2) * (self.duration**2) / self.error)
        )

        pauli_strings = list(hamiltonian.keys())

        num_qubits = len(pauli_strings[0])
        qc = QuantumCircuit(num_qubits)

        term_idxs = list(range(len(list(self.hamiltonian.keys()))))
        probs = [np.abs(weight) / self.norm for weight in self.hamiltonian.values()]
        for idx in range(self.num_terms):
            sample = np.random.choice(term_idxs, p=probs)
            p = list(hamiltonian.keys())[sample]
            temp_qc = exp_pauli_string_to_circ(
                p, self.norm * self.duration / self.num_terms
            )
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
