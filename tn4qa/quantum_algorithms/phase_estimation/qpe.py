from timeit import default_timer
from typing import TypeAlias, Union

import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Operation
from qiskit.circuit.library import QFT
from sparse import SparseArray

from ..backend.base import QuantumBackend
from ..backend.tn_backend import TNQuantumBackend
from ..base import QuantumAlgorithm
from ..hamiltonian_simulation.qdrift import QDriftSimulation
from ..hamiltonian_simulation.trotterisation import TrotterSimulation
from ..result import Result
from ..utils import add_controls, count_qubits, to_quantum_circuit

TypeOptions: TypeAlias = Union[
    QuantumCircuit, Operation, CircuitInstruction, ndarray, SparseArray  # type: ignore
]  # type: ignore


class QPE(QuantumAlgorithm):
    def __init__(
        self,
        unitary: TypeOptions,
        state: TypeOptions,
        num_precision_bits: int,
        backend: QuantumBackend | None = None,
    ) -> "QPE":  # type: ignore
        """
        Constructor for QPE algorithm.

        Args:
            unitary: The unitary operation to estimate the phases for
            state: The input state to QPE
            num_precision_bits: The number of precision bits
            backend: The backend, defaults to TN4QA circuit simulator
        """
        self.num_state_qubits = count_qubits(state)
        self.precision_bits = num_precision_bits

        unitary_circ = to_quantum_circuit(unitary)
        state_circ = to_quantum_circuit(state)
        iqft = QFT(num_precision_bits, inverse=True).decompose()

        qc = QuantumCircuit(self.num_state_qubits + num_precision_bits)
        qc.h(range(num_precision_bits))
        for idx in range(num_precision_bits):
            temp_qc = QuantumCircuit(self.num_state_qubits + num_precision_bits)
            for _ in range(2**idx):
                temp_qc.compose(
                    unitary_circ,
                    qubits=range(
                        num_precision_bits, num_precision_bits + self.num_state_qubits
                    ),
                    inplace=True,
                )
            temp_qc = add_controls(temp_qc, [idx])
            qc.compose(temp_qc, inplace=True)
        qc.compose(
            state_circ,
            qubits=range(
                num_precision_bits, num_precision_bits + self.num_state_qubits
            ),
            inplace=True,
            front=True,
        )
        qc.compose(iqft, qubits=range(num_precision_bits), inplace=True)
        self._circuit = qc
        self.set_backend(backend=backend)
        self.from_ham = False
        self.t = None

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @classmethod
    def from_hamiltonian(
        cls,
        ham: dict,
        state: TypeOptions,
        num_precision_bits: int,
        backend: QuantumBackend | None = None,
        qdrift: bool = False,
    ) -> "QPE":
        """Set up QPE for a Hamiltonian input"""
        if qdrift:
            circ_builder = QDriftSimulation()
        else:
            circ_builder = TrotterSimulation()
        unitary = circ_builder.circuit
        qpe = cls(unitary, state, num_precision_bits, backend)
        qpe.from_ham = True
        norm = np.sum([np.abs(x) for x in ham.values()])
        qpe.t = np.pi / norm
        return qpe

    def measurement_to_phase(self, bitstring: str) -> float:
        """ "Estimate the phase value theta in e^{2*i*pi*theta} given a measured bitstring"""
        phase = 0
        bitstring = bitstring[: self.precision_bits][::-1]
        for b_idx in range(self.precision_bits):
            b = bitstring[b_idx]
            if b == "1":
                phase += 1 / (2 ** (b_idx + 1))
        return phase

    def collect_phase_distribution(self, measurements: dict) -> dict:
        """Estimate the phase values from a set of measurement results"""
        phase_freq_dict = {}
        for k, v in measurements.items():
            phase = self.measurement_to_phase(k)
            if phase in phase_freq_dict:
                phase_freq_dict[phase] += v
            else:
                phase_freq_dict[phase] = v
        return phase_freq_dict

    def get_most_likely_phase(self, phase_freq_dict: dict) -> float:
        """Get the most probable phase"""
        return max(phase_freq_dict, key=phase_freq_dict.get)

    def phase_to_energy(self, phase: float, t: float) -> float:
        """Assuming U = e^{-iHt} we can interpret the phase as an energy eigenvalue"""
        energy = (2 * np.pi * phase) / t
        return energy

    def run(self, num_shots: int = 1024) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value.

        Args:
            num_shots: Number of shots to use per circuit
        """
        start_time = default_timer()
        counts = self.backend.run(self.circuit, shots=num_shots)
        self.phase_freq_dict = self.collect_phase_distribution(counts)
        self.phase = self.get_most_likely_phase(self.phase_freq_dict)
        if self.from_ham:
            self.energy = self.phase_to_energy(self.phase, self.t)
        end_time = default_timer()

        metadata = {
            "algorithm_name": "QPE",
            "num_shots": num_shots,
            "total_runtime": end_time - start_time,
            "phase_freq_dict": self.phase_freq_dict,
        }
        if self.backend is not None:
            metadata["backend_name"] = self.backend.name
            metadata["backend_coupling_map"] = self.backend.coupling_map
            metadata["backend_basis_gates"] = self.backend.basis_gates
            metadata["backend_num_qubits"] = self.backend.num_qubits

        if self.from_ham:
            result = self.energy
        else:
            result = self.phase

        result = Result(
            result=result,
            measurements=counts,
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
