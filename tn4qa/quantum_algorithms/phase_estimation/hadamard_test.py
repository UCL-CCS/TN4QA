import copy
from timeit import default_timer
from typing import TypeAlias, Union

from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Operation
from sparse import SparseArray

from ..backend.base import QuantumBackend
from ..backend.tn_backend import TNQuantumBackend
from ..base import QuantumAlgorithm
from ..result import Result
from ..utils import add_controls, count_qubits, to_quantum_circuit

TypeOptions: TypeAlias = Union[
    QuantumCircuit, Operation, CircuitInstruction, ndarray, SparseArray  # type: ignore
]


class HadamardTest(QuantumAlgorithm):
    def __init__(
        self,
        unitary: TypeOptions,
        state: TypeOptions,
        backend: QuantumBackend | None = None,
    ) -> "HadamardTest":  # type: ignore
        """Constructor

        Args:
            unitary: The unitary operator to estimate the expectation value with respect to
            state: The state to estimate the expectation value with respect to
            backend: The backend to use, default to TN4QA circuit simulator
        """
        self.num_state_qubits = count_qubits(state)

        state_circ = to_quantum_circuit(state)
        unitary_circ = to_quantum_circuit(unitary)

        qc = QuantumCircuit(self.num_state_qubits + 1)
        qc.compose(
            unitary_circ, qubits=range(1, self.num_state_qubits + 1), inplace=True
        )
        qc = add_controls(qc, [0])
        qc.compose(
            state_circ,
            qubits=range(1, self.num_state_qubits + 1),
            inplace=True,
            front=True,
        )
        self._circuit = qc
        self.set_backend(backend=backend)

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    def get_real_counts(self, num_shots: int) -> dict[str, int]:
        """Get counts for real part of expectation value"""
        qc_real = copy.deepcopy(self.circuit)
        qc_h = QuantumCircuit(self.num_state_qubits + 1)
        qc_h.h(0)
        qc_real.compose(qc_h, inplace=True, front=True)
        qc_real.h(0)
        real_counts = self.backend.run(qc_real, num_shots)
        return real_counts

    def get_imag_counts(self, num_shots: int) -> dict[str, int]:
        """Get counts for real part of expectation value"""
        qc_imag = copy.deepcopy(self.circuit)
        qc_prep = QuantumCircuit(self.num_state_qubits + 1)
        qc_prep.h(0)
        qc_prep.sdg(0)
        qc_imag.compose(qc_prep, inplace=True, front=True)
        qc_imag.h(0)
        imag_counts = self.backend.run(qc_imag, num_shots)
        return imag_counts

    def calculate_exp_value(
        self, real_counts: dict, imag_counts: dict, num_shots: int
    ) -> complex:
        """Calculate the expectation value given counts"""
        counts0_real = 0
        counts0_imag = 0
        for bitstring, count in real_counts.items():
            ancilla_bit = bitstring[0]
            if ancilla_bit == "0":
                counts0_real += count
        for bitstring, count in imag_counts.items():
            ancilla_bit = bitstring[0]
            if ancilla_bit == "0":
                counts0_imag += count
        real_part = 2 * (counts0_real / num_shots) - 1
        imag_part = 2 * (counts0_imag / num_shots) - 1
        return real_part + 1.0j * imag_part

    def run(self, num_shots: int = 1024) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value.

        Args:
            num_shots: Number of shots to use per circuit
        """
        start_time = default_timer()
        real_counts = self.get_real_counts(num_shots)
        imag_counts = self.get_imag_counts(num_shots)
        self.expectation_value = self.calculate_exp_value(
            real_counts, imag_counts, num_shots
        )
        end_time = default_timer()

        metadata = {
            "algorithm_name": "Hadamard Test",
            "num_shots": num_shots,
            "total_runtime": end_time - start_time,
        }
        if self.backend is not None:
            metadata["backend_name"] = self.backend.name
            metadata["backend_coupling_map"] = self.backend.coupling_map
            metadata["backend_basis_gates"] = self.backend.basis_gates
            metadata["backend_num_qubits"] = self.backend.num_qubits
        result = Result(
            result=self.expectation_value,
            measurements=None,
            parameters=None,
            metadata=metadata,
        )
        return result

    def set_backend(self, backend: QuantumBackend | None) -> None:
        """Attach a QuantumBackend instance for execution."""
        if backend is None:
            backend = TNQuantumBackend()
        self.backend = backend
        return
