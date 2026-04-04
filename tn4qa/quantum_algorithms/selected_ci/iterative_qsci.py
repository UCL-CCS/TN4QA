from timeit import default_timer
from typing import Callable

import numpy as np

from ...dmrg import DMRG
from ...mpo import MatrixProductOperator
from ...mps import MatrixProductState
from ..backend.base import QuantumBackend
from ..backend.qiskit_simulator import AerSimulator
from ..base import QuantumAlgorithm
from ..result import Result
from .controlled_time_evolved_qsci import ControlledTimeEvolvedQSCI
from .time_evolved_qsci import TimeEvolvedQSCI


class IterativeQSCI(QuantumAlgorithm):
    def __init__(
        self,
        hamiltonian: dict,
        niters: int,
        method: str,
        method_args: dict,
        dmrg_max_bond: int = 2,
        dmrg_maxiter: int = 10,
        scoring_function: Callable | None = None,
    ) -> "IterativeQSCI":
        self.method = method
        if method == "TE":
            self.hamiltonian = self.sanitize_dict(hamiltonian)
        elif method == "CTE":
            hamiltonian = self.sanitize_dict(hamiltonian)
            hamiltonian = self.normalise_hamiltonian(hamiltonian)
            self.hamiltonian = self.rescale_hamiltonian(hamiltonian)
        else:
            raise ValueError("Method unknown")
        self.method_args = method_args
        self.initial_reference_state = self.run_dmrg()
        self.niters = niters
        self.dmrg_max_bond = dmrg_max_bond
        self.dmrg_maxiter = dmrg_maxiter
        self.scoring_function = scoring_function

        self.all_results = []
        self.all_energies = []
        self.all_subspace_sizes = []
        self.all_subspaces = []
        self.all_groundstates = []

    def sanitize_dict(self, d: dict[str, complex | float]) -> dict[str, float]:
        return {
            k: float(v.real) if isinstance(v, complex) else float(v)
            for k, v in d.items()
        }

    def normalise_hamiltonian(self, d: dict[str, float]) -> dict[str, float]:
        norm = np.sum([np.abs(x) for x in d.values()])
        return {k: v / norm for k, v in d.items()}

    def rescale_hamiltonian(self, d: dict[str, float]) -> dict[str, float]:
        num_qubits = len(list(d.keys())[0])
        d["I" * num_qubits] = d.get("I" * num_qubits, 0) + 1.0
        return {k: v * np.pi / 2 for k, v in d.items()}

    def run_dmrg(self) -> tuple[MatrixProductState, float]:
        """Run DMRG"""
        dmrg = DMRG(self.hamiltonian, max_mps_bond=self.dmrg_max_bond)
        dmrg.run(maxiter=self.dmrg_maxiter)
        return dmrg.mps, dmrg.energy

    def run_one_shot(
        self, num_shots: int, subspace_size: int, reference_state: MatrixProductState
    ):
        if self.method == "TE":
            qsci = TimeEvolvedQSCI(
                self.hamiltonian, reference_state, **self.method_args
            )
        elif self.method == "CTE":
            qsci = ControlledTimeEvolvedQSCI(
                self.hamiltonian, reference_state, **self.method_args
            )
        result = qsci.run(num_shots, subspace_size)
        self.all_results.append(result)
        energy, gs = result.result
        subspace = result.metadata["subspace"]
        subspace_size = result.metadata["actual_subspace_size"]
        self.all_energies.append(energy)
        self.all_groundstates.append(gs)
        self.all_subspace_sizes.append(subspace_size)
        self.all_subspaces.append(subspace)

        return subspace, gs

    def calculate_discovery_coefficient(
        self, hamiltonian: MatrixProductOperator, bitstring: MatrixProductState
    ):
        h_bitstring = bitstring.apply_mpo(hamiltonian)
        ip = h_bitstring.compute_inner_product(h_bitstring).real
        exp_val = np.abs(h_bitstring.compute_inner_product(bitstring)) ** 2
        return np.sqrt(ip - exp_val)

    def calculate_scoring_function(
        self, iteration_number: int, amplitude: float, discovery_coefficient: float
    ):
        if self.scoring_function is None:
            return (iteration_number / self.niters) * amplitude + (
                1 - (iteration_number / self.niters)
            ) * discovery_coefficient
        else:
            return self.scoring_function(
                iteration_number, amplitude, discovery_coefficient
            )

    def prepare_reference_state(
        self, iteration_number: int, subspace: list[str], gs: np.ndarray
    ) -> MatrixProductState:
        discovery_coeffs = []
        ham = MatrixProductOperator.from_hamiltonian(self.hamiltonian)
        for idx in range(len(subspace)):
            bitstring = MatrixProductState.from_bitstring(subspace[idx])
            d = self.calculate_discovery_coefficient(ham, bitstring)
            discovery_coeffs.append(d)
        total_d = sum([d**2 for d in discovery_coeffs])
        normalised_discovery_coeffs = [d / np.sqrt(total_d) for d in discovery_coeffs]

        scores = []
        for idx in range(len(subspace)):
            f = self.calculate_scoring_function(
                iteration_number, gs[idx], normalised_discovery_coeffs[idx]
            )
            scores.append(f)

        total_s = sum([f**2 for f in scores])
        weights = [f / np.sqrt(total_s) for f in scores]

        reference_state = MatrixProductState.from_bitstring(subspace[0])
        reference_state.multiply_by_constant(weights[0])
        for idx in range(1, len(subspace)):
            temp_state = MatrixProductState.from_bitstring(subspace[idx])
            temp_state.multiply_by_constant(weights[idx])
            reference_state = reference_state + temp_state

        return reference_state

    def run(self, num_shots: int, subspace_size: int):
        start_timer = default_timer()
        reference_state = None
        shots_per_iteration = int(num_shots / self.niters)
        for iteration in range(self.niters):
            if iteration == 0:
                reference_state = self.initial_reference_state
            subspace, gs = self.run_one_shot(
                shots_per_iteration, subspace_size, reference_state
            )
            reference_state = self.prepare_reference_state(iteration, subspace, gs)
        end_timer = default_timer()

        metadata = {
            "algorithm_name": "IterativeQSCI",
            "num_shots": num_shots,
            "max_subspace_size": subspace_size,
            "all_energies": self.all_energies,
            "all_subspaces": self.all_subspaces,
            "all_subspace_sizes": self.all_subspace_sizes,
            "all_groundstates": self.all_groundstates,
            "total_runtime": end_timer - start_timer,
        }
        if self.backend is not None:
            metadata["backend_name"] = self.backend.name
            metadata["backend_coupling_map"] = self.backend.coupling_map
            metadata["backend_basis_gates"] = self.backend.basis_gates
            metadata["backend_num_qubits"] = self.backend.num_qubits

        result = self.all_energies[-1], self.all_groundstates[-1]

        result = Result(
            result=result,
            measurements=None,
            parameters=None,
            metadata=metadata,
        )
        return result

    def set_backend(self, backend: QuantumBackend | None) -> None:
        """Attach a QuantumBackend instance for execution."""
        if backend is None:
            backend = AerSimulator()
        self.backend = backend
        return
