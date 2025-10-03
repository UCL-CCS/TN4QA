import copy
from typing import Callable

import numpy as np
from numpy import ndarray

from tn4qa.qi_cost_functions import (
    cost_function_dict_to_purity_mpo,
    cost_function_to_dict,
)

from ..circuit_simulator import CircuitSimulator
from ..dmrg import DMRG
from ..mpo import MatrixProductOperator
from ..mps import MatrixProductState
from ..quantum_algorithms.hamiltonian_simulation.trotterisation import TrotterSimulation
from ..tn import TensorNetwork


class ActiveSpaceSelection:
    def __init__(self, hamiltonian: dict[str, complex], coeff_matrix: ndarray):
        """Constructor

        Args:
            hamiltonian: System Hamiltonian
            coeff_matrix: HF coefficient matrix of shape (N, N)
        """
        self.hamiltonian = hamiltonian
        self.num_spin_orbitals = coeff_matrix.shape[1]
        self.num_orbitals = int(self.num_spin_orbitals / 2)
        self.coeff_matrix = coeff_matrix
        self.all_costs = []

    def run(
        self, num_active_orbitals: int, cost_function: Callable, **kwargs
    ) -> ndarray:
        """
        Perform active space selection by optimising a unitary transformation of the orbital coefficients.

        Args:
            num_active_orbitals [int]: Number of active orbitals to select
            cost_function [Callable]: The cost function to use for orbital optimisation
            kwargs: Valid arguments to provide -
                dmrg_max_mps_bond [int]: maximum bond dimension for DMRG, default 8
                dmrg_method [str]: either "one-site" or "two-site", default "two-site"
                dmrg_convergence_threshold [float]: convergence threshold for DMRG, default 1e-9
                dmrg_initial_state [MatrixProductState]: an initial MPS state for DMRG, default random MPS
                dmrg_maxiter: maximum number of sweeps to perform in DMRG, default 10
                cost_function_decay_power [float]: Required parameter for cost_mutual_info_decay, default 2.0
                cost_function_max_bond [int]: Maximum bond dimension for cost function MPO
                optimisation_max_bond [int]: Maximum bond dimension for optimisation
                optimisation_learning_rate [float]: LR for gradient descent optimisation
                optimisation_maxiter [int]: Maximum iterations for gradient descent optimisation
                optimisation_grad_tolerance [float]: Gradient convergence threshold for gradient descent optimisation
                optimisation_cost_tolerance [float]: Cost convergence threshold for gradient descent optimisation


        Returns:
            Transformed coefficient matrix with optimal active orbitals
        """
        function_args = kwargs
        N = self.num_spin_orbitals
        assert (
            self.coeff_matrix.shape[1] == N
        ), "Number of columns must be twice the number of rows"
        assert (
            self.coeff_matrix.shape[0] == N / 2
        ), "Number of columns must be twice the number of rows"

        # Write the Hamiltonian and perfrom DMRG to get the initial state |psi>_C
        print("Start DMRG")
        max_mps_bond = function_args.get("dmrg_max_mps_bond", 2)
        method = function_args.get("dmrg_method", "two-site")
        convergence_threshold = function_args.get("dmrg_convergence_threshold", 1e-9)
        initial_state = function_args.get("dmrg_initial_state", None)
        maxiter = function_args.get("dmrg_maxiter", 10)
        psi_C = self.run_dmrg(
            hamiltonian=self.hamiltonian,
            max_mps_bond=max_mps_bond,
            method=method,
            convergence_threshold=convergence_threshold,
            initial_state=initial_state,
            maxiter=maxiter,
        )

        # Pauli Mapping Lookup
        self.param_to_pauli_dict = self.build_param_lookup(self.num_spin_orbitals)

        # Cost function to MPO
        print("Start building cost MPO")
        decay_power = function_args.get("cost_function_decay_power", 2.0)
        cost_max_bond = function_args.get("cost_function_max_bond", None)
        cost_mpo = self.build_cost_function_mpo(
            cost_function=cost_function, decay_power=decay_power, max_bond=cost_max_bond
        )

        # Run gradient descent optimisation to find optimal theta
        print("Start optimisation")
        theta_init = np.zeros((N**2,), dtype=float)  # Initial guess for theta
        opt_max_bond = function_args.get("rotation_mpo_max_bond", 16)
        opt_lr = function_args.get("optimisation_learning_rate", 0.01)
        opt_max_iter = function_args.get("optimisation_maxiter", 100)
        opt_grad_tol = function_args.get("optimisation_grad_tolerance", 1e-18)
        opt_cost_tol = function_args.get("optimisation_cost_tolerance", 1e-12)
        self.theta_opt = self.gradient_descent(
            theta_init,
            self.param_to_pauli_dict,
            psi_C,
            cost_mpo,
            opt_max_bond,
            opt_lr,
            opt_max_iter,
            opt_grad_tol,
            opt_cost_tol,
        )

        # Exponentiate K to get a unitary U = exp(K)
        K_opt = self.vector_to_antihermitian(self.theta_opt)
        U = self.exponentiate_K(K_opt)

        # Apply U to the input coefficient matrix, returning the transformed coefficient matrix (the new basis)
        self.transformed_coeff_matrix = self.coeff_matrix @ U

        return self.transformed_coeff_matrix

    def run_dmrg(
        self,
        hamiltonian: dict[str, complex],
        max_mps_bond: int | None,
        method: str,
        convergence_threshold: float,
        initial_state: MatrixProductState,
        maxiter: int,
    ) -> MatrixProductState:
        """Run DMRG to get an approximate groundstate in the initial MO basis"""
        dmrg = DMRG(
            hamiltonian,
            max_mps_bond=max_mps_bond,
            method=method,
            convergence_threshold=convergence_threshold,
            initial_state=initial_state,
        )
        _, psi = dmrg.run(maxiter=maxiter)
        return psi

    def build_cost_function_mpo(
        self, cost_function: Callable, decay_power: float, max_bond: int | None = None
    ) -> MatrixProductOperator:
        """Build the cost function as an MPO"""
        d = cost_function_to_dict(
            cost_function, num_orbitals=self.num_orbitals, decay_power=decay_power
        )
        mpo = cost_function_dict_to_purity_mpo(self.num_spin_orbitals, d, max_bond)
        return mpo

    def vector_to_antihermitian(self, theta: ndarray) -> ndarray:
        """
        Converts a real vector of length N^2 into an anti-Hermitian matrix K ∈ C^{N x N}.

        Diagonal entries are pure imaginary: iθ
        Off-diagonal: K[p,q] = a + ib, K[q,p] = -a + ib
        """
        norbs = self.num_spin_orbitals
        assert len(theta) == self.num_spin_orbitals**2, "theta must have length N^2"

        K = np.zeros((norbs, norbs), dtype=complex)
        idx = 0

        # Fill diagonals: all imaginary
        for i in range(norbs):
            K[i, i] = 1j * theta[idx]
            idx += 1

        # Fill upper triangle, set lower triangle with Hermitian conjugate
        for i in range(norbs):
            for j in range(i + 1, norbs):
                real = theta[idx]
                imag = theta[idx + 1]
                K[i, j] = real + 1j * imag
                K[j, i] = -real + 1j * imag  # = -conj(K[i,j])
                idx += 2

        return K

    # Map parameter index → (p,q) orbital pairs
    def param_to_indices(self, N: int, k: int):
        """
        Map theta index k into the orbital indices (p,q) it controls.
        Diagonal: (p,p)
        Off-diagonal: (p,q) and (q,p), tagged with 'real'/'imag'
        """
        # Case 1: diagonal
        if k < N:
            return [(k, k, "diag")]

        # Case 2: off-diagonal
        j = k - N
        pair_index, offset = divmod(j, 2)

        # find (p,q) for this pair_index
        count = 0
        for p in range(N):
            for q in range(p + 1, N):
                if count == pair_index:
                    if offset == 0:
                        return [(p, q, "real")]
                    else:
                        return [(p, q, "imag")]
                count += 1

        raise ValueError("Index out of range")

    # Map θ_i → {PauliString: coeff}
    # Build Pauli dictionary for each θ_i
    def build_param_lookup(self, N: int) -> dict[int, dict[str, complex]]:
        """
        Build mapping {theta index i: {PauliString: coeff}}.
        Each theta corresponds to either:
        - diagonal: (1/2)(I - Z_p)
        - off-diagonal real: (1/2)(X_p X_q + Y_p Y_q) * Z-string
        - off-diagonal imag: (1/2)(X_p Y_q - Y_p X_q) * Z-string
        """
        lookup = {}

        for i in range(N**2):
            term_dict = {}
            for p, q, tag in self.param_to_indices(N, i):
                if tag == "diag":
                    # (i/2)(I - Z_p)
                    term_dict["I" * N] = 0.5
                    Zstr = ["I"] * N
                    Zstr[p] = "Z"
                    term_dict["".join(Zstr)] = -0.5
                else:
                    # Build Z-string between p and q
                    Zbase = ["I"] * N
                    start, stop = min(p, q), max(p, q)
                    for k in range(start + 1, stop):
                        Zbase[k] = "Z"

                    if tag == "real":
                        # (1/2)(X_p X_q + Y_p Y_q) * Z-string
                        Xstr = Zbase.copy()
                        Ystr = Zbase.copy()
                        Xstr[p], Xstr[q] = "X", "X"
                        Ystr[p], Ystr[q] = "Y", "Y"
                        term_dict["".join(Xstr)] = 0.5
                        term_dict["".join(Ystr)] = 0.5

                    elif tag == "imag":
                        # (i/2)(X_p Y_q - Y_p X_q) * Z-string
                        XYstr = Zbase.copy()
                        YXstr = Zbase.copy()
                        XYstr[p], XYstr[q] = "X", "Y"
                        YXstr[p], YXstr[q] = "Y", "X"
                        term_dict["".join(XYstr)] = 0.5
                        term_dict["".join(YXstr)] = -0.5

            lookup[i] = term_dict

        return lookup

    def calculate_gradients(
        self,
        theta: np.ndarray,
        pauli_lookup: dict,
        mpo: MatrixProductOperator,
        mps: MatrixProductState,
        max_bond: int | None,
    ) -> dict[int, float]:
        mpo = copy.deepcopy(mpo)
        mps = copy.deepcopy(mps)
        gradients = {}

        pauli_ham_dict = {}
        for idx, d in pauli_lookup.items():
            d = {key: val * theta[idx] for key, val in d.items()}
            pauli_ham_dict.update(d)

        num_params = len(list(pauli_lookup.keys()))
        num_spinorbs = int(np.sqrt(num_params))

        # Create exp(Σ_{pq} K_{pq} a†_p a_q) |mps>
        trotter_circ = TrotterSimulation(pauli_ham_dict, 1.0, num_steps=1)
        sim = CircuitSimulator(trotter_circ.circuit, input_state=mps)
        rotated_state = sim.run(max_bond_dimension=max_bond)

        # Calculate gradient for each theta_k
        for k, pauli_dict in reversed(list(pauli_lookup.items())):
            # Create MPO for gradient specific term
            complex_pauli_dict = {
                key: complex(1j * val) for key, val in pauli_dict.items()
            }
            grad_mpo = MatrixProductOperator.from_hamiltonian(complex_pauli_dict)

            # Build full TN for gradient calculation
            rotated_state_dag = copy.deepcopy(rotated_state)
            rotated_state_dag.dagger()
            rotated_state_copy = copy.deepcopy(rotated_state)
            rotated_state_dag_copy = copy.deepcopy(rotated_state_dag)

            rotated_state.set_default_indices("A", "B", 1)
            mpo.set_default_indices("C", "B", "D", 1)
            grad_mpo.set_default_indices("E", "D", "F", 1)
            rotated_state_dag.set_default_indices("G", "F", 1)
            rotated_state_copy.set_default_indices("H", "B", num_spinorbs + 1)
            rotated_state_dag_copy.set_default_indices("I", "D", num_spinorbs + 1)

            all_tensors = (
                rotated_state.tensors
                + mpo.tensors
                + grad_mpo.tensors
                + rotated_state_dag.tensors
                + rotated_state_copy.tensors
                + rotated_state_dag_copy.tensors
            )
            full_tn = TensorNetwork(all_tensors)
            grad = full_tn.contract_entire_network()
            gradients[k] = 4 * grad.real

            # Update rotated_state
            reversed_pauli_dict = {
                key: -1.0 * val * theta[k]
                for key, val in reversed(list(pauli_dict.items()))
            }
            trotter_circ = TrotterSimulation(reversed_pauli_dict, 1.0, num_steps=1)
            sim = CircuitSimulator(trotter_circ.circuit, input_state=rotated_state)
            rotated_state = sim.run(max_bond_dimension=max_bond)

            # Update MPO
            pauli_dict_theta = {key: val * theta[k] for key, val in pauli_dict.items()}
            trotter_circ = TrotterSimulation(pauli_dict_theta, 1.0, num_steps=1)
            mpo.evolve_by_quantum_circuit(trotter_circ.circuit)

        return gradients

    def calculate_cost(
        self,
        theta: ndarray,
        pauli_lookup: dict,
        mpo: MatrixProductOperator,
        mps: MatrixProductState,
        max_bond: int | None,
    ) -> float:
        """Optimisation cost function.

        Args;
            theta: Paramter list for K
            mpo: QI cost function MPO
            mps: Groundstate approximation from DMRG

        Returns:
            < MPS | (exp(Σ_{pq} K_{pq} a†_p a_q))† MPO exp(Σ_{pq} K_{pq} a†_p a_q) | MPS >
        """
        pauli_ham_dict = {}
        for idx, d in pauli_lookup.items():
            d = {key: val * theta[idx] for key, val in d.items()}
            pauli_ham_dict.update(d)

        # Create exp(Σ_{pq} K_{pq} a†_p a_q) |mps>
        trotter_circ = TrotterSimulation(pauli_ham_dict, 1.0, num_steps=1)
        sim = CircuitSimulator(trotter_circ.circuit, input_state=mps)
        rotated_state = sim.run(max_bond_dimension=max_bond)

        # Calculate cost
        rotated_state_doubled = rotated_state.to_two_copy_mps()
        cost = rotated_state_doubled.compute_expectation_value(mpo)
        return cost.real

    # def optimise_K(
    #     self,
    #     theta_init: ndarray,
    #     mpo: MatrixProductOperator,
    #     mps: MatrixProductState,
    #     max_bond: int | None = None,
    # ):
    #     """
    #     Run BFGS optimisation over K to minimise optimisation_cost.

    #     Args:
    #         theta_init: Initial guess for theta
    #         mpo: QI cost function MPO
    #         mps: Groundstate approximation from DMRG

    #     Returns:
    #         Optimal real-valued parameter vector θ defining anti-Hermitian K
    #     """

    #     result = minimize(
    #         self.optimisation_cost,
    #         theta_init,
    #         args=(mpo, mps, max_bond),
    #         method="COBYLA",
    #         options={"disp": True, "maxiter": 50},
    #     )
    #     print("Optimisation result:", result.x)
    #     return result.x

    # ----- Gradient Descent Loop -----
    def gradient_descent(
        self,
        theta_init: np.ndarray,
        pauli_lookup: dict,
        mps: MatrixProductState,
        mpo: MatrixProductOperator,
        max_bond: int | None,
        lr: float,
        max_iters: int,
        grad_tol: float,
        cost_tol: float,
    ) -> np.ndarray:
        """
        Simple gradient descent loop.
        Arguments:
        theta_init : initial array of thetas
        pauli_lookup : map from theta index to Pauli strings
        mps : state for optimisation
        mpo : Cost function MPO
        max_bond : Maximum bond dimension
        lr         : learning rate
        max_iters  : maximum iterations
        tol        : convergence threshold tolerance
        Returns:
        theta      : optimised parameters
        """
        theta = theta_init.copy()

        for iter in range(max_iters):
            self.all_costs.append(
                self.calculate_cost(theta, pauli_lookup, mpo, mps, max_bond)
            )

            grad_dict = self.calculate_gradients(
                theta, pauli_lookup, mpo, mps, max_bond
            )
            grad = np.array(list(grad_dict.values()), dtype=float)

            grad_norm = np.linalg.norm(grad)
            if grad_norm < grad_tol:
                print(f"Converged at iteration {iter}, grad_norm={grad_norm:.3e}")
                break

            cost_diff = np.abs(self.all_costs[-1] - self.all_costs[-2])
            if iter > 10:
                if cost_diff < cost_tol:
                    print(f"Converged at iteration {iter}, cost_diff={cost_diff:.3e}")
                    break

            theta -= lr * grad
            if iter % 2 == 0 or iter == max_iters - 1:
                print(
                    f"Iteration number: {iter:3d} grad_norm={grad_norm:.3e} cost={self.all_costs[-1]}"
                )

        return theta

    def exponentiate_K(self, K: ndarray) -> ndarray:
        """
        Compute U = exp(K) using eigendecomposition, where K is anti-Hermitian.

        Args:
        K: Anti-Hermitian matrix of shape (N, N)

        Returns:
        U = exp(K): a unitary matrix
        """
        assert K.shape[0] == K.shape[1], "K must be square"
        assert np.allclose(K + K.conj().T, 0), "K must be anti-Hermitian"

        # Eigendecomposition: K = V D V^{-1}
        eigvals, eigvecs = np.linalg.eig(K)

        # Compute exp(K) = V exp(D) V^{-1)
        exp_D = np.diag(np.exp(eigvals))
        V_inv = np.linalg.inv(eigvecs)
        U = eigvecs @ exp_D @ V_inv
        return U
