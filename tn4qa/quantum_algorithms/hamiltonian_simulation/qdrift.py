from timeit import default_timer

import numpy as np
from qiskit import QuantumCircuit

from ..backend.base import QuantumBackend
from ..backend.tn_backend import TNQuantumBackend
from ..base import QuantumAlgorithm
from ..result import Result
from ..utils import calculate_exp_val_stochastic, exp_pauli_string_to_circ


class QDriftSimulation(QuantumAlgorithm):
    """
    Perform Hamiltonian simulation using qDRIFT
    """

    def __init__(
        self,
        hamiltonian: dict[str, float],
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

        self._circuit = self.build_circuit(
            self.hamiltonian, self.norm, self.num_terms, self.duration
        )
        self.set_backend(backend)

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    def build_circuit(
        self, ham: dict[str, float], norm: float, num_terms: int, duration: float
    ) -> QuantumCircuit:
        pauli_strings = list(ham.keys())

        num_qubits = len(pauli_strings[0])
        qc = QuantumCircuit(num_qubits)

        term_idxs = list(range(len(list(ham.keys()))))
        probs = [np.abs(weight) / norm for weight in ham.values()]
        for _ in range(num_terms):
            sample = np.random.choice(term_idxs, p=probs)
            p = list(ham.keys())[sample]
            sign = 1 if ham[p] >= 0.0 else -1
            temp_qc = exp_pauli_string_to_circ(p, norm * duration * sign / num_terms)
            qc.compose(temp_qc, inplace=True)

        return qc

    def run(
        self,
        num_shots: int = 1024,
        num_circuits: int | None = None,
        observable: dict | None = None,
    ) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value."""
        start_time = default_timer()
        counts = {}
        shots_per_circ = int(num_shots / num_circuits)
        for _ in range(num_circuits):
            self._circuit = self.build_circuit(
                self.hamiltonian, self.norm, self.num_terms, self.duration
            )
            bs = self.backend.run(self.circuit, shots=shots_per_circ)
            for b, count in bs.items():
                if b in counts:
                    counts[b] += count
                else:
                    counts[b] = count
        if observable is None:
            result = None
        else:
            result = calculate_exp_val_stochastic(
                self.build_circuit,
                observable,
                self.backend,
                num_shots,
                self.hamiltonian,
                self.norm,
                self.num_terms,
                self.duration,
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
