from collections.abc import Callable
from timeit import default_timer
from typing import Union

import matplotlib.pyplot as plt
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import Optimizer

from ..backend.base import QuantumBackend
from ..base import QuantumAlgorithm
from ..result import Result
from .ansatz_circuits import (
    hea_ansatz,
    number_preserving_ansatz,
    pauli_two_design_ansatz,
)
from .classical_optimisers import (
    adam_optimiser,
    bfgs_optimiser,
    cobyla_optimiser,
    qnspsa_optimiser,
)


class VQEAlgorithm(QuantumAlgorithm):
    def __init__(
        self,
        hamiltonian: dict,
        num_electrons: int | None = None,
        max_iterations_vqe: int = 1e3,
        convergence_threshold: float = 1e-6,
        initial_points: ndarray = None,
        ansatz: Union[QuantumCircuit, str] | None = None,
        optimiser: Union[Optimizer, str] | None = None,
        estimator: Estimator | None = None,
        backend: QuantumBackend | None = None,
    ) -> None:
        """
        Constructor

        Args:
            hamiltonian: a qubit Hamiltonian given as a dict
            num_electrons: Required if using number preserving ansatz
            max_iterations_vqe: Maximum number of iterations
            convergence_threshold: Convergence threshold for early exit
            initial_points: Initial starting points
            ansatz: Ansatz given as a QuantumCircuit or a string (number_preserving_ansatz, hardware_efficient_ansatz, pauli_two_design_ansatz)
            optimiser: Optimiser given as an Optimiser or a string (QNSPSA, ADAM, BFGS, COBYLA)
            estimator: Estimator given as an Estimator or None
            backend: QuantumBackend, currently only Qiskit simulation
        """

        if isinstance(hamiltonian, dict):
            self.num_qubits = len(list(hamiltonian.keys())[0])
        else:
            self.num_qubits = len(hamiltonian[0])
        self.hamiltonian = hamiltonian
        self.num_electrons = num_electrons
        self.optimisation_index = 0
        self.max_iterations_vqe = max_iterations_vqe
        self.convergence_threshold = convergence_threshold
        self.initial_points = initial_points

        self.clear_optimisation_dict()
        self.clear_warm_starting_dict()

        self.set_backend(backend=backend)

        self.set_ansatz(ansatz=ansatz)
        self.set_optimiser(optimiser=optimiser)
        self.set_estimator(estimator=estimator)

        self.set_callback()
        self.driver = self.vqe_driver()

    @property
    def circuit(self) -> QuantumCircuit:
        return self.ansatz

    def set_estimator(self, estimator: Estimator | None = None) -> None:
        """Set the Estimator

        Args:
            estimator: The Estimator, optional, defaults to Estimator()
        """
        if not estimator:
            self.estimator = Estimator()
        else:
            self.estimator = estimator
        return

    def set_ansatz(self, ansatz: QuantumCircuit | str | None = None) -> None:
        """Set ansatz circuit

        Args:
            ansatz: QuantumCircuit, string identifier or None, defaults to number_preserving_ansatz
        """
        if not ansatz or ansatz == "number_preserving_ansatz":
            qc1 = QuantumCircuit(self.num_qubits)
            qc1.x(range(self.num_electrons))
            qc2 = number_preserving_ansatz(self.num_qubits, 3, "linear")
            qc = qc1.compose(qc2)
            self.ansatz = qc
        elif ansatz == "hardware_efficient_ansatz":
            qc = hea_ansatz(self.num_qubits)
            self.ansatz = qc
        elif ansatz == "pauli_two_design_ansatz":
            qc = pauli_two_design_ansatz(self.num_qubits)
            self.ansatz = qc
        elif isinstance(ansatz, str):
            print(
                "Ansatz string not recognised. Must be one of 'hea_ansatz', 'number_preserving_ansatz', 'pauli_two_design_ansatz', 'uccsd_ansatz'"
            )
        else:
            assert isinstance(ansatz, QuantumCircuit)
            self.ansatz = ansatz
        return

    def set_optimiser(self, optimiser: Optimizer | str | None = None) -> None:
        """Set optimiser

        Args:
            optimiser: Optimiser or string identifier or None, defaults to QNSPSA
        """
        self.optimisation_index = 0
        if not optimiser or optimiser == "QNSPSA":
            self.optimiser = qnspsa_optimiser(
                self.ansatz,
                self.max_iterations_vqe,
                opt_dict=self.optimisation_dict,
                index=self.optimisation_index,
            )
        elif optimiser == "ADAM":
            self.optimiser = adam_optimiser(
                self.max_iterations_vqe,
                opt_dict=self.optimisation_dict,
                index=self.optimisation_index,
            )
        elif optimiser == "BFGS":
            self.optimiser = bfgs_optimiser(
                self.max_iterations_vqe,
                opt_dict=self.optimisation_dict,
                index=self.optimisation_index,
            )
        elif optimiser == "COBYLA":
            self.optimiser = cobyla_optimiser(
                self.max_iterations_vqe,
                self.convergence_threshold,
                opt_dict=self.optimisation_dict,
                index=self.optimisation_index,
            )
        elif isinstance(optimiser, str):
            print(
                "Optimiser string not recognised. Must be one of 'ADAM', 'COBYLA', 'QNSPSA', 'BFGS'"
            )
        else:
            assert isinstance(optimiser, Optimizer)
            self.optimiser = optimiser
        return

    def set_callback(self, callback: Callable | None = None) -> None:
        """Set callback

        Args:
            callback: Function or None
        """
        if not callback:

            def default_callback(i, a, f, _):
                # print(i, a, f)
                self.optimisation_dict["optimisation_number"].append(i)
                self.optimisation_dict["optimisation_parameters"].append(a)
                self.optimisation_dict["optimisation_value"].append(f)
                return

            self.callback = default_callback
        else:
            self.callback = callback
        return

    def clear_optimisation_dict(self) -> None:
        """Reset optimisation dictionary"""
        self.optimisation_dict = {
            "optimisation_number": [],
            "optimisation_parameters": [],
            "optimisation_value": [],
        }
        return

    def clear_warm_starting_dict(self) -> None:
        """Reset warm starting dictionary"""
        self.warm_starting_dict = {
            "optimisation_number": [],
            "optimisation_parameters": [],
            "optimisation_value": [],
        }
        return

    def vqe_driver(self) -> VQE:
        """Set the VQE driver"""
        driver = VQE(
            self.estimator,
            self.ansatz,
            self.optimiser,
            initial_point=self.initial_points,
            callback=self.callback,
        )
        return driver

    def plot_convergence(self) -> None:
        """Plot convergence data"""
        plt.plot(
            self.optimisation_dict["optimisation_number"],
            self.optimisation_dict["optimisation_value"],
        )
        plt.show()
        return

    def run(self) -> Result:
        """Run the full algorithm pipeline. Returns result object."""
        pauli_list = list(self.hamiltonian.keys())
        coeffs = list(self.hamiltonian.values())
        observable = SparsePauliOp(pauli_list, coeffs)
        self.clear_optimisation_dict()
        start_time = default_timer()
        result = self.driver.compute_minimum_eigenvalue(observable)
        end_time = default_timer()
        self.minimum_eigenvalue = result.eigenvalue
        metadata = {
            "algorithm_name": "VQE",
            "num_parameters": len(
                self.optimisation_dict["optimisation_parameters"][-1]
            ),
            "num_iterations": self.optimisation_dict["optimisation_number"][-1],
            "total_runtime": end_time - start_time,
        }
        if self.backend is not None:
            metadata["backend_name"] = self.backend.name
            metadata["backend_coupling_map"] = self.backend.coupling_map
            metadata["backend_basis_gates"] = self.backend.basis_gates
            metadata["backend_num_qubits"] = self.backend.num_qubits
        result = Result(
            result=self.minimum_eigenvalue,
            measurements=None,
            parameters=self.optimisation_dict["optimisation_parameters"][-1],
            metadata=metadata,
        )
        return result

    def set_backend(self, backend: QuantumBackend | None = None) -> None:
        """Attach a QuantumBackend instance for execution.

        Args:
            backend: Currently only supports Qiskit backends
        """
        self.backend = backend
        return
