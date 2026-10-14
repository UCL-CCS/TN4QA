import heapq
from timeit import default_timer
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit

from ...mpo import MatrixProductOperator
from ..backend.base import QuantumBackend
from ..backend.qiskit_simulator import QiskitSimulatorBackend
from ..base import QuantumAlgorithm
from ..result import Result
from .diagonalisation import subspace_energy
from .spin_symmetry import (
    detect_spin_orbital_order,
    spin_occupations,
    spin_orbital_partition,
)


class QSCI(QuantumAlgorithm):
    def __init__(
        self,
        circuits: list[QuantumCircuit],
        backend: QuantumBackend | None,
        hamiltonian: dict[str, complex],
        num_electrons: int | None = None,
        known_important_configurations: list[str] = [],
        known_unimportant_configurations: list[str] = [],
        postprocessing_functions: list[Callable] | None = None,
        postprocessing_args: list[dict] | None = None,
        num_alpha: int | None = None,
        num_beta: int | None = None,
        spin_orbital_order: str | None = None,
        verify_spin: bool = True,
        admission_threshold: int = 1,
    ) -> "QSCI":
        """
        Constructor for QSCI class.
        """
        self.hamiltonian = hamiltonian
        self.num_electrons = num_electrons
        self.num_alpha = num_alpha
        self.num_beta = num_beta
        self.spin_orbital_order = spin_orbital_order
        self.verify_spin = verify_spin
        self.hamiltonian_mpo = MatrixProductOperator.from_hamiltonian(self.hamiltonian)
        self.circuits = circuits
        self.set_backend(backend=backend)
        self.energy = None
        self.important_configurations = known_important_configurations
        self.unimportant_configurations = known_unimportant_configurations
        self.postprocessing_functions = postprocessing_functions
        self.postprocessing_args = postprocessing_args
        self.admission_threshold = admission_threshold

    @property
    def circuit(self) -> QuantumCircuit:
        """Get the circuit from the algorithm"""
        return self.circuits

    def execute_circuits(self, shots_per_circuit: int) -> dict:
        """Execute circuits and collect results"""
        all_counts = {}
        for circ in self.circuits:
            counts = self.backend.run(circ, shots_per_circuit)
            for b, c in counts.items():
                all_counts[b] = all_counts.get(b, 0) + c
        return all_counts

    def postprocessing(self, counts: dict[str, int], args: list[dict] | None):
        if self.postprocessing_functions is None:
            return counts
        else:
            assert len(self.postprocessing_functions) == len(self.postprocessing_args)
            for idx in range(len(self.postprocessing_functions)):
                func = self.postprocessing_functions[idx]
                func_args = args[idx]
                if func_args is None:
                    counts = func(counts)
                else:
                    counts = func(counts, **args)
            return counts

    def symmetry_verification(
        self,
        counts: dict[str, int],
        particle_number: int | None = None,
        num_alpha: int | None = None,
        num_beta: int | None = None,
    ) -> dict:
        """Discard samples that violate a symmetry the Hamiltonian conserves.

        Checks, in order of strength:
            * the total electron number, if ``particle_number`` is given;
            * the alpha and beta counts separately -- equivalently the total
            together with S_z -- if ``num_alpha`` and ``num_beta`` are known.

        The alpha/beta check subsumes the total, so passing both is consistent
        rather than redundant.  S^2 is not checked; it is not diagonal in the
        computational basis and so is not a property of a single sample.

        The rejected fraction is recorded on ``self.symmetry_rejections`` so a
        caller can report it.  A large alpha/beta rejection rate with a small
        total-number rejection rate is the signature this method exists for:
        the samples have the right electron count and the wrong S_z.
        """
        submitted = sum(counts.values())
        new_counts = {k: v for k, v in counts.items() if v > self.admission_threshold}
        submitted = sum(new_counts.values())
        rejected_number = rejected_spin = 0

        if particle_number is not None:
            kept = {
                k: v for k, v in new_counts.items() if k.count("1") == particle_number
            }
            rejected_number = submitted - sum(kept.values())
            new_counts = kept

        if (
            getattr(self, "verify_spin", True)
            and num_alpha is not None
            and num_beta is not None
            and new_counts
        ):
            order = getattr(self, "spin_orbital_order", None)
            if order is None:
                order = detect_spin_orbital_order(self.hamiltonian)
                self.spin_orbital_order = order
            if order is not None:
                n = len(next(iter(new_counts)))
                alpha, beta = spin_orbital_partition(n, order)
                before = sum(new_counts.values())
                new_counts = {
                    k: v
                    for k, v in new_counts.items()
                    if spin_occupations(k, alpha, beta) == (num_alpha, num_beta)
                }
                rejected_spin = before - sum(new_counts.values())

        self.symmetry_rejections = dict(
            submitted=submitted,
            rejected_particle_number=rejected_number,
            rejected_spin=rejected_spin,
            surviving=sum(new_counts.values()),
            spin_orbital_order=getattr(self, "spin_orbital_order", None),
        )
        return new_counts

    def gather_samples(self, cr_counts: dict, k: int) -> list[str]:
        """Collect the (at most) k most frequent samples to form the selected subspace"""
        top_samples = heapq.nlargest(k, cr_counts, key=cr_counts.get)
        return top_samples

    def diagonalisation(self, samples: list[str]):
        energy, gs = subspace_energy(self.hamiltonian, samples)
        return energy, gs

    def run(self, num_shots: int, subspace_size: int) -> Result:
        """Run the full algorithm pipeline. Returns result object or final value."""
        start_time = default_timer()
        shots_per_circuit = int(num_shots / len(self.circuits))
        counts = self.execute_circuits(shots_per_circuit)
        pp_counts = self.postprocessing(counts, self.postprocessing_args)
        sv_counts = self.symmetry_verification(
            pp_counts, self.num_electrons, self.num_alpha, self.num_beta
        )
        samples = self.gather_samples(sv_counts, subspace_size)
        samples = [s for s in samples if s not in self.unimportant_configurations]
        samples = list(set(samples) | set(self.important_configurations))
        self.energy, groundstate_vec = self.diagonalisation(samples)
        end_time = default_timer()

        circ_depths = [qc.depth() for qc in self.circuits]
        avg_depth = np.mean(circ_depths)
        # The mean alone is not comparable between a run whose circuits all
        # sit at one evolution time and a run whose circuits are a ladder:
        # the ladder's mean describes a circuit nobody executes, while the
        # coherence requirement is set by the DEEPEST circuit.  Both are
        # reported so the caller can pick the one its claim needs.
        max_depth = float(np.max(circ_depths)) if circ_depths else float("nan")

        metadata = {
            "algorithm_name": "QSCI",
            "num_shots": num_shots,
            "max_subspace_size": subspace_size,
            "actual_subspace_size": len(samples),
            "subspace": samples,
            "avg_circuit_depth": avg_depth,
            "max_circuit_depth": max_depth,
            "circuit_depths": [int(d) for d in circ_depths],
            "num_circuits": len(self.circuits),
            "counts": counts,
            "pp_counts": pp_counts,
            "sv_counts": sv_counts,
            "total_runtime": end_time - start_time,
        }
        if self.backend is not None:
            metadata["backend_name"] = self.backend.name
            metadata["backend_coupling_map"] = self.backend.coupling_map
            metadata["backend_basis_gates"] = self.backend.basis_gates
            metadata["backend_num_qubits"] = self.backend.num_qubits

        result = self.energy, groundstate_vec

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
            backend = QiskitSimulatorBackend()
        self.backend = backend
        return
