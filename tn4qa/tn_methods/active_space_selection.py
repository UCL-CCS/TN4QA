import copy
from typing import Callable

import numpy as np
import scipy
import sparse
from numpy import ndarray

from tn4qa.qi_cost_functions import (
    cost_function_dict_to_purity_mpo,
    cost_function_to_dict,
)

from ..dmrg import DMRG
from ..mpo import MatrixProductOperator
from ..mps import MatrixProductState
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
            cost_function=cost_function,
            decay_power=decay_power,
            max_bond=cost_max_bond,
            num_active_orbitals=num_active_orbitals,
        )

        # Run gradient descent optimisation to find optimal theta
        print("Start optimisation")
        theta_init = np.zeros((N**2,), dtype=float)  # Initial guess for theta
        opt_max_bond = function_args.get("rotation_mpo_max_bond", 16)
        opt_lr = function_args.get("optimisation_learning_rate", 0.01)
        opt_max_iter = function_args.get("optimisation_maxiter", 100)
        opt_grad_tol = function_args.get("optimisation_grad_tolerance", 1e-16)
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
        self,
        cost_function: Callable,
        decay_power: float,
        max_bond: int | None = None,
        num_active_orbitals: int | None = None,
    ) -> MatrixProductOperator:
        """Build the cost function as an MPO"""
        d = cost_function_to_dict(
            cost_function,
            num_orbitals=self.num_orbitals,
            decay_power=decay_power,
            num_active_orbitals=num_active_orbitals,
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
        - off-diagonal imag: (1/2)(X_p X_q + Y_p Y_q) * Z-string
        - off-diagonal real: (1/2)(X_p Y_q - Y_p X_q) * Z-string
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

                    if tag == "imag":
                        # (1/2)(X_p X_q + Y_p Y_q) * Z-string
                        Xstr = Zbase.copy()
                        Ystr = Zbase.copy()
                        Xstr[p], Xstr[q] = "X", "X"
                        Ystr[p], Ystr[q] = "Y", "Y"
                        term_dict["".join(Xstr)] = 0.5
                        term_dict["".join(Ystr)] = 0.5

                    elif tag == "real":
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
        only_real_params: bool = True,
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

        ##########
        # Contract bottom half of gradient TN
        ##########

        # Create exp(Σ_{pq} K_{pq} a†_p a_q) |mps>
        rotated_state = mps
        for pauli_string, coeff in pauli_ham_dict.items():
            if coeff == 0.0:
                continue
            temp_mpo = MatrixProductOperator.from_pauli_exponential(pauli_string, coeff)
            rotated_state = rotated_state.apply_mpo(temp_mpo, max_bond=max_bond)
            rotated_state.normalise()

        # And its inverse
        rotated_state_dag = copy.deepcopy(rotated_state)
        rotated_state_dag.dagger()

        # Create and contract a TN
        mpo.set_default_indices("E", "D", "B")
        rotated_state.set_default_indices("A", "B", num_spinorbs + 1)
        rotated_state_dag.set_default_indices("C", "D", num_spinorbs + 1)
        required_mpo_tensors = [
            mpo.tensors[x] for x in range(num_spinorbs, 2 * num_spinorbs)
        ]
        tn = TensorNetwork(
            rotated_state.tensors + required_mpo_tensors + rotated_state_dag.tensors
        )
        result = tn.contract_entire_network()  # Rank 1 tensor
        rotated_state.set_default_indices("A", "B")
        rotated_state_dag.set_default_indices("C", "D")

        # Create new MPO
        required_mpo_tensors = [mpo.tensors[x] for x in range(num_spinorbs)]
        required_mpo_tensors[-1].data = sparse.einsum(
            "abcd,b->acd", required_mpo_tensors[-1].data, result.data
        )
        new_mpo_arrays = [required_mpo_tensors[x].data for x in range(num_spinorbs)]
        new_mpo = MatrixProductOperator.from_arrays(new_mpo_arrays)
        new_mpo.set_default_indices("E", "D", "F")

        ##########
        # Contract left half of gradient TN
        ##########

        # Create and contract TN
        left_mps = rotated_state_dag.apply_mpo(new_mpo, max_bond)
        left_mps.set_default_indices("C", "D")

        ##########
        # Loop through each parameter and calculate gradient
        ##########

        # Do the last parameter
        last_k = num_params - 2 if only_real_params else num_params - 1
        last_pauli_dict = pauli_lookup[last_k]
        last_ham = {key: complex(0.5j * val) for key, val in last_pauli_dict.items()}
        grad_mpo = MatrixProductOperator.from_hamiltonian(last_ham)
        last_rotated_state = rotated_state.apply_mpo(grad_mpo, max_bond)
        res = last_rotated_state.compute_inner_product(left_mps)
        gradients[last_k] = 4 * res.real

        # Loop through the rest
        for k, pauli_dict in reversed(pauli_lookup.items()):
            if k == last_k:
                continue

            if only_real_params:
                if k < num_spinorbs:
                    gradients[k] = 0.0
                    continue
                elif k % 2 == 1:
                    gradients[k] = 0.0
                    continue

            # Create the Pauli string MPO
            ham = {key: complex(0.5j * val) for key, val in pauli_dict.items()}
            grad_mpo = MatrixProductOperator.from_hamiltonian(ham)

            # Compute the rotated state
            current_rotated_state = mps
            for l, p_dict in pauli_lookup.items():
                for ps, x in p_dict.items():
                    temp_mpo = MatrixProductOperator.from_pauli_exponential(ps, x)
                    current_rotated_state.apply_mpo(temp_mpo, max_bond)
                    current_rotated_state.normalise()
                if l == k:
                    current_rotated_state.apply_mpo(grad_mpo)

            # Create and contract TN
            res = current_rotated_state.compute_inner_product(left_mps)
            gradients[k] = 4 * res.real

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
        rotated_state = mps
        for pauli_string, coeff in pauli_ham_dict.items():
            if coeff == 0.0:
                continue
            temp_mpo = MatrixProductOperator.from_pauli_exponential(pauli_string, coeff)
            rotated_state = rotated_state.apply_mpo(temp_mpo, max_bond=max_bond)
            rotated_state.normalise()

        # Calculate cost
        rotated_state_doubled = rotated_state.to_two_copy_mps()
        cost = rotated_state_doubled.compute_expectation_value(mpo)
        return cost.real

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
        only_real_params: bool = True,
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
                theta, pauli_lookup, mpo, mps, max_bond, only_real_params
            )
            grad = np.array([grad_dict[i] for i in range(len(theta))], dtype=float)

            grad_norm = np.linalg.norm(grad)
            if grad_norm < grad_tol:
                print(f"Converged at iteration {iter}, grad_norm={grad_norm:.3e}")
                break

            if iter >= 1:
                cost_diff = np.abs(self.all_costs[-1] - self.all_costs[-2])
            if iter > 100:
                if cost_diff < cost_tol:
                    print(f"Converged at iteration {iter}, cost_diff={cost_diff:.3e}")
                    break

            theta -= lr * grad
            # if iter % 5 == 0 or iter == max_iters - 1:
            print(
                f"Iteration number: {iter:3d} grad_norm={grad_norm:.3e} cost={self.all_costs[-1]}"
            )

        return theta

    def exponentiate_K(self, K: ndarray) -> ndarray:
        """
        Compute U = exp(-K) using eigendecomposition, where K is anti-Hermitian.

        Args:
        K: Anti-Hermitian matrix of shape (N, N)

        Returns:
        U = exp(-K): a unitary matrix
        """
        assert K.shape[0] == K.shape[1], "K must be square"
        assert np.allclose(K + K.conj().T, 0), "K must be anti-Hermitian"

        U = scipy.linalg.expm(-1.0 * K)
        return U
