from timeit import default_timer
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit

from ...dmrg import DMRG
from ...mpo import MatrixProductOperator
from ...mps import MatrixProductState
from ..backend.base import QuantumBackend
from ..backend.qiskit_simulator import QiskitSimulatorBackend
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
        num_electrons: int | None = None,
        dmrg_max_bond: int = 2,
        dmrg_maxiter: int = 10,
        scoring_function: Callable | None = None,
        backend: QuantumBackend | None = None,
    ) -> "IterativeQSCI":
        self.method = method
        if method == "TE":
            self.hamiltonian = self.sanitize_dict(hamiltonian)
        elif method == "CTE":
            self.hamiltonian = self.sanitize_dict(hamiltonian)
            # hamiltonian = self.normalise_hamiltonian(hamiltonian)
            # self.hamiltonian = self.rescale_hamiltonian(hamiltonian)
        else:
            raise ValueError("Method unknown")
        self.method_args = method_args
        self.dmrg_max_bond = dmrg_max_bond
        self.dmrg_maxiter = dmrg_maxiter
        self.initial_reference_state = self.run_dmrg()
        self.nelec = num_electrons
        if self.nelec:
            self.initial_reference_state = MatrixProductState.from_hf_state(
                self.initial_reference_state.num_sites, self.nelec
            )
        self.niters = niters
        self.scoring_function = scoring_function
        self.backend = self.set_backend(backend)

        self.circuits = []

        self.all_results = []
        self.all_energies = []
        self.all_subspace_sizes = []
        self.all_subspaces = []
        self.all_groundstates = []
        self.important_configurations = []
        self.unimportant_configurations = []
        self.all_circuit_depths = []

        self.cached_discover_coefficients = {}

    @property
    def circuit(self) -> QuantumCircuit:
        """Get the circuit from the algorithm"""
        return self.circuits

    def sanitize_dict(self, d: dict[str, complex | float]) -> dict[str, float]:
        return {
            k: float(v.real) if isinstance(v, complex) else float(v)
            for k, v in d.items()
        }

    # def normalise_hamiltonian(self, d: dict[str, float]) -> dict[str, float]:
    #     self.norm = np.sum([np.abs(x) for x in d.values()])
    #     return {k: v / self.norm for k, v in d.items()}

    # def rescale_hamiltonian(self, d: dict[str, float]) -> dict[str, float]:
    #     num_qubits = len(list(d.keys())[0])
    #     d["I" * num_qubits] = d.get("I" * num_qubits, 0) + 1.0
    #     return {k: v * np.pi / 2 for k, v in d.items()}

    # def rescale_energy(self, energy_prime):
    #         return self.norm * ((2 / np.pi) * energy_prime - 1)

    def run_dmrg(self) -> MatrixProductState:
        """Run DMRG"""
        dmrg = DMRG(self.hamiltonian, max_mps_bond=self.dmrg_max_bond)
        _, gs = dmrg.run(nsweeps=self.dmrg_maxiter)
        return gs

    def run_one_shot(
        self, num_shots: int, subspace_size: int, reference_state: MatrixProductState
    ):
        if self.method == "TE":
            qsci = TimeEvolvedQSCI(
                self.hamiltonian,
                reference_state,
                backend=self.backend,
                known_important_configurations=self.important_configurations,
                known_unimportant_configurations=self.unimportant_configurations,
                **self.method_args,
            )
        elif self.method == "CTE":
            qsci = ControlledTimeEvolvedQSCI(
                self.hamiltonian,
                reference_state,
                backend=self.backend,
                known_important_configurations=self.important_configurations,
                known_unimportant_configurations=self.unimportant_configurations,
                **self.method_args,
            )
        self.circuits = qsci.circuit
        result = qsci.run(num_shots, subspace_size)
        self.all_results.append(result)
        energy, gs = result.result
        subspace = result.metadata["subspace"]
        subspace_size = result.metadata["actual_subspace_size"]
        self.all_energies.append(energy)
        self.all_groundstates.append(gs)
        self.all_subspace_sizes.append(subspace_size)
        self.all_subspaces.append(subspace)
        self.all_circuit_depths.append(result.metadata["avg_circuit_depth"])

        for sidx in range(len(subspace)):
            sample = subspace[sidx]
            amp = gs[sidx]
            if np.abs(amp) ** 2 > 1e-16:
                self.important_configurations.append(sample)
            else:
                self.unimportant_configurations.append(sample)
        self.important_configurations = list(set(self.important_configurations))
        self.unimportant_configurations = list(set(self.unimportant_configurations))

        return subspace, gs

    def calculate_discovery_coefficient(self, bitstring: MatrixProductState):
        if bitstring in self.cached_discover_coefficients:
            return self.cached_discover_coefficients[bitstring]
        ham_mpo = MatrixProductOperator.from_hamiltonian(self.hamiltonian)
        h_bitstring = bitstring.apply_mpo(ham_mpo)
        ip = h_bitstring.compute_inner_product(h_bitstring).real
        exp_val = np.abs(h_bitstring.compute_inner_product(bitstring)) ** 2
        diff = max(ip - exp_val, 0)
        dc = np.sqrt(diff)
        self.cached_discover_coefficients[bitstring] = dc
        return dc

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
        for idx in range(len(subspace)):
            bitstring = MatrixProductState.from_bitstring(subspace[idx])
            d = self.calculate_discovery_coefficient(bitstring)
            discovery_coeffs.append(d)
        # total_d = sum([d**2 for d in discovery_coeffs])
        # if total_d != 0:
        #     normalised_discovery_coeffs = [
        #         d / np.sqrt(total_d) for d in discovery_coeffs
        #     ]
        # else:
        #     normalised_discovery_coeffs = discovery_coeffs

        scores = []
        for idx in range(len(subspace)):
            f = self.calculate_scoring_function(
                iteration_number, gs[idx], discovery_coeffs[idx]
            )
            scores.append(f)

        total_s = sum([f**2 for f in scores])
        if total_s != 0:
            weights = [f / np.sqrt(total_s) for f in scores]
        else:
            weights = [1 / np.sqrt(len(subspace)) for _ in scores]

        d = {subspace[idx]: weights[idx] for idx in range(len(subspace))}

        reference_state = MatrixProductState.from_bitstring_dict(d, 8)
        # reference_state.compress(2)
        reference_state.normalise()

        return reference_state

    def run(self, num_shots: int, subspace_size: int):
        start_timer = default_timer()
        reference_state = None
        shots_per_iteration = int(num_shots / self.niters)
        for iteration in range(self.niters):
            print("iteration number", iteration + 1)
            if iteration == 0:
                reference_state = self.initial_reference_state
            _, gs = self.run_one_shot(
                shots_per_iteration, subspace_size, reference_state
            )
            if len(self.important_configurations) > 0:
                reference_state = self.prepare_reference_state(
                    iteration, self.important_configurations, gs
                )
        end_timer = default_timer()

        metadata = {
            "algorithm_name": "IterativeQSCI",
            "num_shots": num_shots,
            "max_subspace_size": subspace_size,
            "all_energies": self.all_energies,
            "all_subspaces": self.all_subspaces,
            "all_subspace_sizes": self.all_subspace_sizes,
            "all_groundstates": self.all_groundstates,
            "all_circuit_depths": self.all_circuit_depths,
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
            backend = QiskitSimulatorBackend()
        self.backend = backend
        return
