import copy
from typing import Callable

import numpy as np
import scipy
import sparse
from numpy import ndarray

from tn4qa.qi_cost_functions import (
    cost_function_dict_to_callable,
    cost_function_dict_to_purity_mpo,
    cost_function_to_dict,
)

from ..dmrg import DMRG
from ..mpo import MatrixProductOperator
from ..mps import MatrixProductState
from ..tn import TensorNetwork


class MolecularOrbitalOptimisation:
    def __init__(
        self,
        hamiltonian: dict[str, complex],
        coeff_matrix: ndarray,
        restricted: bool = True,
        nelec: int | None = None,
    ):
        """Constructor

        Args:
            hamiltonian: System Hamiltonian
            coeff_matrix: HF coefficient matrix of shape (N,N) if restricted, and (2,N,N) if unrestricted
            restricted: If true assumes RHF coeff matrix, else UHF
            nelec: Number of electrons in the system, used to initialise DMRG with HF state
        """
        self.hamiltonian = hamiltonian
        self.num_spin_orbitals = int(2 * coeff_matrix.shape[1])
        self.num_orbitals = int(self.num_spin_orbitals / 2)
        self.all_costs = []
        self.grad_norms = []
        self.cost_function_callable = None
        self.restricted = restricted
        self.energy_minimisation = False
        self.nelec = nelec

        def interleave_spin_blocks(A, B):
            N = A.shape[0]
            M = np.zeros((2 * N, 2 * N), dtype=A.dtype)

            M[0::2, 0::2] = A  # alpha-alpha block
            M[1::2, 1::2] = B  # beta-beta block

            return M

        if restricted:
            self.coeff_matrix = interleave_spin_blocks(coeff_matrix, coeff_matrix)
        else:
            self.coeff_matrix = interleave_spin_blocks(coeff_matrix[0], coeff_matrix[1])

    def run(
        self,
        num_active_orbitals: int,
        cost_function: Callable | dict[str, complex],
        **kwargs,
    ) -> ndarray:
        """
        Perform active space selection by optimising a unitary transformation of the orbital coefficients.

        Args:
            num_active_orbitals [int]: Number of active orbitals to select
            cost_function [Callable]: The cost function to use for orbital optimisation
            kwargs: Valid arguments to provide -
                dmrg_max_mps_bond [int]: maximum bond dimension for DMRG, default 8
                dmrg_maxiter: maximum number of sweeps to perform in DMRG, default 10
                cost_function_decay_power [float]: Required parameter for cost_mutual_info_decay, default 2.0
                cost_function_max_bond [int]: Maximum bond dimension for cost function MPO
                optimisation_method [str]: Optimisation method, either "gradient_descent" or "quasi_newton"
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
        ), "Number of columns must be the number of rows"
        assert (
            self.coeff_matrix.shape[0] == N
        ), "Number of rows must be the number of spin orbitals"

        # Write the Hamiltonian and perfrom DMRG to get the initial state |psi>_C
        print("Start DMRG")
        max_mps_bond = function_args.get("dmrg_max_mps_bond", 2)
        maxiter = function_args.get("dmrg_maxiter", 10)
        psi_C = self.run_dmrg(
            hamiltonian=self.hamiltonian,
            max_mps_bond=max_mps_bond,
            maxiter=maxiter,
        )

        # Pauli Mapping Lookup
        if self.restricted:
            self.param_to_pauli_dict = self.build_param_lookup_restricted(
                self.num_orbitals
            )
        else:
            self.param_to_pauli_dict = self.build_param_lookup_unrestricted(
                self.num_spin_orbitals
            )

        # Cost function to MPO
        print("Start building cost MPO")
        decay_power = function_args.get("cost_function_decay_power", 2.0)
        cost_max_bond = function_args.get("cost_function_max_bond", None)
        active_orbs = function_args.get("cost_active_orbs", None)
        if isinstance(cost_function, dict):
            cost_mpo = MatrixProductOperator.from_hamiltonian(self.hamiltonian)
            self.energy_minimisation = True
        else:
            cost_mpo = self.build_cost_function_mpo(
                cost_function=cost_function,
                decay_power=decay_power,
                max_bond=cost_max_bond,
                num_active_orbitals=num_active_orbitals,
                active_orbs=active_orbs,
            )

        # Run gradient descent optimisation to find optimal theta
        print("Start optimisation")
        if self.restricted:
            num_params = self.num_orbitals**2
        else:
            num_params = self.num_spin_orbitals**2

        theta_init = np.zeros((num_params,), dtype=float)  # Initial guess for theta
        opt_max_bond = function_args.get("rotation_mpo_max_bond", 8)
        opt_lr = function_args.get("optimisation_learning_rate", 1e-1)
        opt_max_iter = function_args.get("optimisation_maxiter", 100)
        opt_grad_tol = function_args.get("optimisation_grad_tolerance", 1e-8)
        opt_cost_tol = function_args.get("optimisation_cost_tolerance", 1e-12)
        opt_method = function_args.get("optimisation_method", "gradient_descent")
        if opt_method == "gradient_descent":
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
        else:
            self.theta_opt = self.quasi_newton(
                theta_init,
                self.param_to_pauli_dict,
                psi_C,
                cost_mpo,
                opt_max_bond,
                opt_max_iter,
                opt_grad_tol,
                opt_cost_tol,
                energy_minimisation=self.energy_minimisation,
            )

        # Exponentiate K to get a unitary U = exp(K)
        K_opt = self.vector_to_antihermitian(self.theta_opt)
        U = self.exponentiate_K(K_opt)

        def separate_spin_blocks(M):
            A = M[0::2, 0::2].copy()
            B = M[1::2, 1::2].copy()
            return A, B

        # Apply U to the input coefficient matrix, returning the transformed coefficient matrix (the new basis)
        new_coeff_matrix = self.coeff_matrix @ U
        C_alpha, C_beta = separate_spin_blocks(new_coeff_matrix)
        if self.restricted:
            assert np.allclose(C_alpha, C_beta)
            self.transformed_coeff_matrix = C_alpha
        else:
            self.transformed_coeff_matrix = (C_alpha, C_beta)
        return self.transformed_coeff_matrix

    def run_dmrg(
        self,
        hamiltonian: dict[str, complex],
        max_mps_bond: int | None,
        maxiter: int,
    ) -> MatrixProductState:
        """Run DMRG to get an approximate groundstate in the initial MO basis"""
        if self.nelec:
            initial_mps = MatrixProductState.from_hf_state(
                self.num_spin_orbitals, self.nelec
            )
            initial_mps.expand_bond_dimension_list(
                1, list(range(1, initial_mps.num_sites))
            )
        else:
            initial_mps = None
        dmrg = DMRG(hamiltonian, max_mps_bond=max_mps_bond, initial_mps=initial_mps)
        _, psi = dmrg.run(nsweeps=maxiter)
        psi.compress(2)
        return psi

    def build_cost_function_mpo(
        self,
        cost_function: Callable,
        decay_power: float,
        max_bond: int | None = None,
        num_active_orbitals: int | None = None,
        active_orbs: list[int] | None = None,
    ) -> MatrixProductOperator:
        """Build the cost function as an MPO"""
        # Ensure we always pass an explicit list of active orbitals when the cost
        # function expects it (e.g. cost_mutual_info_active_inactive).
        if active_orbs is None:
            if num_active_orbitals is None:
                raise ValueError(
                    "Either active_orbs or num_active_orbitals must be provided "
                    "to build the cost function MPO."
                )
            active_orbs = list(range(num_active_orbitals))

        d = cost_function_to_dict(
            cost_function,
            num_orbitals=self.num_orbitals,
            decay_power=decay_power,
            num_active_orbitals=num_active_orbitals,
            active_orbs=active_orbs,
        )

        def entropy_func(rdm):
            return 1 - np.trace(rdm @ rdm)

        self.cost_function_callable = cost_function_dict_to_callable(d, entropy_func)
        mpo = cost_function_dict_to_purity_mpo(self.num_spin_orbitals, d, max_bond)
        return mpo

    def vector_to_antihermitian(self, theta: ndarray) -> ndarray:
        """
        Converts a real vector of length N^2 into an anti-Hermitian matrix K ∈ C^{N x N}.

        Diagonal entries are pure imaginary: iθ
        Off-diagonal: K[p,q] = a + ib, K[q,p] = -a + ib
        """
        if self.restricted:
            norbs = self.num_orbitals
            assert (
                len(theta) == self.num_orbitals**2
            ), "For restricted case theta must have length norbs^2"
        else:
            norbs = self.num_spin_orbitals
            assert (
                len(theta) == self.num_spin_orbitals**2
            ), "For unrestricted case theta must have length nspinorbs^2"

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

        if self.restricted:
            K = np.kron(K, np.eye(2))  # lift to spin-orbital space

        return K

    # Map parameter index → (p,q) orbital pairs
    def param_to_indices(self, N: int, k: int):
        """
        Map theta index k into the orbital indices (p,q) it controls.
        Diagonal: (p,p)
        Off-diagonal: (p,q) and (q,p), tagged with 'real'/'imag'
        Restricted case: N = num orbitals
        Unrestricted case: N = num spin orbitals
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

    def add_pauli_terms(self, term_dict, p, q, tag, N_spin):
        if tag == "diag":
            term_dict["I" * N_spin] = term_dict.get("I" * N_spin, 0) + 0.5

            Zstr = ["I"] * N_spin
            Zstr[p] = "Z"
            key = "".join(Zstr)
            term_dict[key] = term_dict.get(key, 0) - 0.5

        else:
            Zbase = ["I"] * N_spin
            start, stop = min(p, q), max(p, q)
            for k in range(start + 1, stop):
                Zbase[k] = "Z"

            if tag == "imag":
                Xstr = Zbase.copy()
                Ystr = Zbase.copy()
                Xstr[p], Xstr[q] = "X", "X"
                Ystr[p], Ystr[q] = "Y", "Y"

                key = "".join(Xstr)
                term_dict[key] = term_dict.get(key, 0) + 0.5

                key = "".join(Ystr)
                term_dict[key] = term_dict.get(key, 0) + 0.5

            elif tag == "real":
                XYstr = Zbase.copy()
                YXstr = Zbase.copy()

                XYstr[p], XYstr[q] = "X", "Y"
                YXstr[p], YXstr[q] = "Y", "X"

                key = "".join(XYstr)
                term_dict[key] = term_dict.get(key, 0) + 0.5

                key = "".join(YXstr)
                term_dict[key] = term_dict.get(key, 0) - 0.5

    def build_param_lookup_restricted(self, N_orb):
        N_spin = 2 * N_orb
        lookup = {i: None for i in range(N_orb**2)}

        for i in range(N_orb**2):
            term_dict = {}

            for p, q, tag in self.param_to_indices(N_orb, i):
                # α contribution
                self.add_pauli_terms(term_dict, 2 * p, 2 * q, tag, N_spin)

                # β contribution
                self.add_pauli_terms(term_dict, 2 * p + 1, 2 * q + 1, tag, N_spin)

            lookup[i] = term_dict

        return lookup

    # Map θ_i → {PauliString: coeff}
    # Build Pauli dictionary for each θ_i
    def build_param_lookup_unrestricted(self, N: int) -> dict[int, dict[str, complex]]:
        """
        Build mapping {theta index i: {PauliString: coeff}}.
        Each theta corresponds to either:
        - diagonal: (1/2)(I - Z_p)
        - off-diagonal imag: (1/2)(X_p X_q + Y_p Y_q) * Z-string
        - off-diagonal real: (1/2)(X_p Y_q - Y_p X_q) * Z-string

        Unrestricted case: N = num spin orbitals
        """
        lookup = {i: None for i in range(N**2)}

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

    def calculate_energy_gradients(
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
        gradients = {k: 0.0 for k in range(len(theta))}

        pauli_ham_dict = {}

        for idx, d in pauli_lookup.items():
            for key, val in d.items():
                pauli_ham_dict[key] = pauli_ham_dict.get(key, 0.0) + theta[idx] * val

        num_params = len(list(pauli_lookup.keys()))
        num_spinorbs = int(np.sqrt(num_params))

        # Create exp(Σ_{pq} K_{pq} a†_p a_q) |mps>
        rotated_state = copy.deepcopy(mps)
        for pauli_string, coeff in pauli_ham_dict.items():
            if coeff == 0.0:
                continue
            temp_mpo = MatrixProductOperator.from_pauli_exponential(pauli_string, coeff)
            rotated_state = rotated_state.apply_mpo(temp_mpo, max_bond=max_bond)

        # And its inverse
        rotated_state_dag = copy.deepcopy(rotated_state)
        rotated_state_dag.dagger()

        ##########
        # Loop through each parameter and calculate gradient
        ##########

        # Loop through
        for k, pauli_dict in reversed(pauli_lookup.items()):
            # Don't let alpha and beta spin orbitals mix!
            p, q, _ = self.param_to_indices(self.num_spin_orbitals, k)[0]
            if (p % 2) != (q % 2):
                gradients[k] = 0.0
                continue

            # Keep the rotation orthogonal
            if only_real_params:
                if k < num_spinorbs:
                    gradients[k] = 0.0
                    continue
                elif k % 2 == 1:
                    gradients[k] = 0.0
                    continue

            # Create the Pauli string MPO
            ham = {key: complex(1j * val) for key, val in pauli_dict.items()}
            grad_mpo = MatrixProductOperator.from_hamiltonian(ham)

            # Compute the rotated state
            current_rotated_state = copy.deepcopy(mps)
            for l, p_dict in pauli_lookup.items():
                for ps, x in p_dict.items():
                    temp_mpo = MatrixProductOperator.from_pauli_exponential(ps, x)
                    current_rotated_state = current_rotated_state.apply_mpo(
                        temp_mpo, max_bond
                    )
                if l == k:
                    current_rotated_state = current_rotated_state.apply_mpo(grad_mpo)

            # Create and contract TN
            mpo_dag = copy.deepcopy(mpo)
            mpo_dag.dagger()
            right_mps = rotated_state_dag.apply_mpo(mpo)
            right_mps.dagger()
            res = current_rotated_state.compute_inner_product(right_mps)
            gradients[k] = 2 * res.real

        return gradients

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
        gradients = {k: 0.0 for k in range(len(theta))}

        pauli_ham_dict = {}
        for idx, d in pauli_lookup.items():
            for key, val in d.items():
                pauli_ham_dict[key] = pauli_ham_dict.get(key, 0.0) + theta[idx] * val

        num_spinorbs = self.num_spin_orbitals
        num_orbs = self.num_orbitals

        ##########
        # Contract bottom half of gradient TN
        ##########

        # Create exp(Σ_{pq} K_{pq} a†_p a_q) |mps>
        rotated_state = copy.deepcopy(mps)
        for pauli_string, coeff in pauli_ham_dict.items():
            if coeff == 0.0:
                continue
            temp_mpo = MatrixProductOperator.from_pauli_exponential(pauli_string, coeff)
            rotated_state = rotated_state.apply_mpo(temp_mpo, max_bond=max_bond)

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
        # Contract right half of gradient TN
        ##########

        # Create and contract TN
        new_mpo.dagger()
        right_mps = rotated_state_dag.apply_mpo(new_mpo, max_bond)

        ##########
        # Loop through each parameter and calculate gradient
        ##########

        # Loop through
        for k, pauli_dict in pauli_lookup.items():
            # Keep the rotation orthogonal
            if only_real_params:
                N = num_orbs if self.restricted else num_spinorbs
                if k < N:
                    gradients[k] = 0.0
                    continue
                elif N % 2 == 0 and k % 2 == 1:
                    gradients[k] = 0.0
                    continue
                elif N % 2 == 1 and k % 2 == 0:
                    gradients[k] = 0.0
                    continue

            # Don't let alpha and beta spin orbitals mix!
            N = self.num_orbitals if self.restricted else self.num_spin_orbitals
            p, q, _ = self.param_to_indices(N, k)[0]
            if (p % 2) != (q % 2):
                gradients[k] = 0.0
                continue

            # Create the Pauli string MPO
            ham = {key: complex(1j * val) for key, val in pauli_dict.items()}
            grad_mpo = MatrixProductOperator.from_hamiltonian(ham)

            # Compute the rotated state
            current_rotated_state = copy.deepcopy(mps)
            for l, p_dict in pauli_lookup.items():
                for ps, x in p_dict.items():
                    w = theta[l] * x
                    if w != 0:
                        temp_mpo = MatrixProductOperator.from_pauli_exponential(ps, w)
                        current_rotated_state = current_rotated_state.apply_mpo(
                            temp_mpo, max_bond
                        )
                if l == k:
                    current_rotated_state = current_rotated_state.apply_mpo(grad_mpo)

            # Create and contract TN
            right = copy.deepcopy(right_mps)
            right.dagger()
            res = current_rotated_state.compute_inner_product(right)
            gradients[k] = 4 * res.real

        return gradients

    def calculate_energy(
        self,
        theta: ndarray,
        pauli_lookup: dict,
        mpo: MatrixProductOperator,
        mps: MatrixProductState,
        max_bond: int | None,
    ) -> float:
        pauli_ham_dict = {}

        for idx, d in pauli_lookup.items():
            for key, val in d.items():
                pauli_ham_dict[key] = pauli_ham_dict.get(key, 0.0) + theta[idx] * val

        # Create exp(Σ_{pq} K_{pq} a†_p a_q) |mps>
        rotated_state = copy.deepcopy(mps)
        for pauli_string, coeff in pauli_ham_dict.items():
            if coeff == 0.0:
                continue
            temp_mpo = MatrixProductOperator.from_pauli_exponential(pauli_string, coeff)
            rotated_state = rotated_state.apply_mpo(temp_mpo, max_bond=max_bond)
        rotated_state.normalise()

        # Calculate energy
        cost = rotated_state.compute_expectation_value(mpo)
        return cost.real

    def calculate_cost(
        self,
        theta: ndarray,
        pauli_lookup: dict,
        mps: MatrixProductState,
        mpo: MatrixProductOperator,
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
        rotated_state = copy.deepcopy(mps)
        for k in range(len(theta)):
            pauli_dict = pauli_lookup[k]
            for ps, val in pauli_dict.items():
                coeff = theta[k] * val
                if coeff != 0:
                    temp_mpo = MatrixProductOperator.from_pauli_exponential(ps, coeff)
                    rotated_state = rotated_state.apply_mpo(temp_mpo, max_bond=max_bond)
        rotated_state.normalise()

        # Calculate cost
        doubled_rotated_state = rotated_state.to_two_copy_mps()
        cost = doubled_rotated_state.compute_expectation_value(mpo)
        # cost = self.cost_function_callable(rotated_state)
        return cost.real

    def gradient_descent(
        self,
        theta_init: np.ndarray,
        pauli_lookup: dict,
        mps: MatrixProductState,
        mpo: MatrixProductOperator,
        max_bond: int | None,
        lr: float = 0.01,
        max_iters: int = 200,
        grad_tol: float = 1e-6,
        cost_tol: float = 1e-12,
        only_real_params: bool = True,
        energy_minimisation: bool = False,
        max_norm: float = 0.5,
        min_lr: float = 1e-10,
    ) -> np.ndarray:
        """
        Gradient descent loop with adaptive step size, momentum, and relative cost convergence.
        """

        theta = theta_init.copy()
        num_params = len(theta)

        # ----- Compute allowed parameter mask once -----
        allowed = np.ones(num_params, dtype=bool)
        N = self.num_orbitals if self.restricted else self.num_spin_orbitals

        for k in range(num_params):
            if only_real_params:
                if k < N:
                    allowed[k] = False
                elif N % 2 == 0 and k % 2 == 1:
                    allowed[k] = False
                elif N % 2 == 1 and k % 2 == 0:
                    allowed[k] = False

            p, q, _ = self.param_to_indices(N, k)[0]
            if (p % 2) != (q % 2):
                allowed[k] = False

        # Functions to expand/reduce theta arrays
        def expand_theta(t):
            full = np.zeros_like(theta_init)
            full[allowed] = t
            return full

        def cost(t):
            full_theta = expand_theta(t)
            if energy_minimisation:
                return self.calculate_energy(
                    full_theta, pauli_lookup, mpo, mps, max_bond
                )
            else:
                return self.calculate_cost(full_theta, pauli_lookup, mps, mpo, max_bond)

        def grad(t):
            full_theta = expand_theta(t)
            if energy_minimisation:
                grad_dict = self.calculate_energy_gradients(
                    full_theta, pauli_lookup, mpo, mps, max_bond, only_real_params
                )
            else:
                grad_dict = self.calculate_gradients(
                    full_theta, pauli_lookup, mpo, mps, max_bond, only_real_params
                )
            grad_array = np.zeros(num_params)
            for k, v in grad_dict.items():
                grad_array[k] = v
            return grad_array[allowed]

        # ----- Initialise theta and momentum -----
        theta_opt = theta[allowed]

        self.all_costs = []
        prev_cost = cost(theta_opt)
        self.all_costs.append(prev_cost)

        lr_current = lr

        for iter in range(max_iters):
            g = grad(theta_opt)
            grad_norm = np.linalg.norm(g)
            self.grad_norms.append(grad_norm)

            if grad_norm < grad_tol:
                print(f"[Converged] Iter {iter}, grad_norm={grad_norm:.3e}")
                break

            if grad_norm > max_norm:
                g = g * (max_norm / grad_norm)
            theta_trial = theta_opt - lr_current * g
            cost_old = prev_cost
            cost_new = cost(theta_trial)

            if iter >= 0:  # changed this from iter > 1
                if cost_new < cost_old:
                    # Accept step
                    theta_opt = theta_trial
                    prev_cost = cost_new
                    lr_current *= 1.05  # grow slightly
                    step_accepted = True
                else:
                    # Reject step
                    lr_current *= 0.5  # shrink
                    step_accepted = False
            else:
                theta_opt = theta_trial
                prev_cost = cost_new
                step_accepted = True

            self.all_costs.append(prev_cost)

            status = "ACCEPT" if step_accepted else "REJECT"
            print(
                f"Iter {iter:3d} | grad_norm={grad_norm:.3e} | cost={cost_new:.6e} "
                f"| lr={lr_current:.3e} | {status}"
            )

            if step_accepted:
                cost_diff = abs(cost_old - cost_new)
                if cost_diff < cost_tol and iter > 5:
                    print(
                        f"[Converged] Iter {iter}, relative cost change={cost_diff:.3e}"
                    )
                    break

            if lr_current < min_lr:
                print("LR too small. Resetting")
                theta_opt += np.random.normal(0, 1e-4, size=theta_opt.shape)
                lr_current = lr * 0.5

        # Expand back to full theta array
        new_theta = expand_theta(theta_opt)
        return new_theta

    def quasi_newton(
        self,
        theta_init: np.ndarray,
        pauli_lookup: dict,
        mps: MatrixProductState,
        mpo: MatrixProductOperator,
        max_bond: int | None,
        max_iters: int,
        grad_tol: float,
        cost_tol: float,
        only_real_params: bool = True,
        energy_minimisation: bool = False,
    ) -> np.ndarray:
        """
        Quasi-Newton descent loop.
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
        allowed = np.array([True] * len(theta), dtype=bool)
        for k in range(len(theta)):
            if only_real_params:
                N = self.num_orbitals if self.restricted else self.num_spin_orbitals
                if k < N:
                    allowed[k] = False
                elif N % 2 == 0 and k % 2 == 1:
                    allowed[k] = False
                elif N % 2 == 1 and k % 2 == 0:
                    allowed[k] = False
        # No mixing of alpha-beta spins!
        for k in range(len(theta)):
            N = self.num_orbitals if self.restricted else self.num_spin_orbitals
            p, q, _ = self.param_to_indices(N, k)[0]
            if (p % 2) != (q % 2):
                allowed[k] = False

        theta_opt = theta[allowed]

        def expand_theta(t):
            full = np.zeros_like(theta_init)
            full[allowed] = t
            return full

        def cost(t):
            full_theta = expand_theta(t)
            if energy_minimisation:
                cost = self.calculate_energy(
                    full_theta, pauli_lookup, mpo, mps, max_bond
                )
            else:
                cost = self.calculate_cost(full_theta, pauli_lookup, mps, mpo, max_bond)
            return cost

        def grad(t):
            full_theta = expand_theta(t)
            if energy_minimisation:
                grad = self.calculate_energy_gradients(
                    full_theta, pauli_lookup, mpo, mps, max_bond, only_real_params
                )
            else:
                grad = self.calculate_gradients(
                    full_theta, pauli_lookup, mpo, mps, max_bond, only_real_params
                )
            grad_array = np.zeros(len(theta_init))
            for k, v in grad.items():
                grad_array[k] = v

            grad = grad_array[allowed]
            return grad

        # print("BEGIN")
        # t_copy = theta_opt.copy()
        # analytic_grads = grad(t_copy)
        # eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-15, 1e-20, 1e-25]
        # for ep in eps:
        #     for i in range(len(theta_opt)):
        #         g_analytic = analytic_grads[i]
        #         t_copy[i] += ep
        #         c_plus = cost(t_copy)
        #         t_copy[i] -= 2*ep
        #         c_minus = cost(t_copy)
        #         g_fd = (c_plus - c_minus) / (2*ep)
        #         print(i, ep, g_analytic, g_fd, g_analytic - g_fd)

        iteration = 0

        def callback(xk):
            nonlocal iteration
            full_theta = expand_theta(xk)
            if energy_minimisation:
                c = self.calculate_energy(full_theta, pauli_lookup, mpo, mps, max_bond)
                g = self.calculate_energy_gradients(
                    full_theta, pauli_lookup, mpo, mps, max_bond, only_real_params
                )
            else:
                c = self.calculate_cost(full_theta, pauli_lookup, mps, mpo, max_bond)
                g = self.calculate_gradients(
                    full_theta, pauli_lookup, mpo, mps, max_bond, only_real_params
                )
            g = np.array(list(g.values()))[allowed]
            grad_norm = np.linalg.norm(g)
            print(f"iter={iteration:3d} cost={c:.12f} |grad|={grad_norm:.6e}")
            iteration += 1

        num_active = np.sum(allowed)
        assert num_active > 0, "No parameters left to optimize!"

        bounds = [(-0.5, 0.5)] * len(theta_opt)
        opt = scipy.optimize.minimize(
            cost,
            theta_opt,
            method="L-BFGS-B",
            jac=grad,
            callback=callback,
            bounds=bounds,
            options={
                "maxiter": max_iters,
                "ftol": cost_tol,
                "gtol": grad_tol,
                "maxcor": 20,
                "maxls": 40,
            },
        )

        print(opt.success)
        print(opt.message)
        print(opt.nit)
        print(opt.nfev)
        print(opt.fun)
        print(np.linalg.norm(opt.jac))

        final_theta = expand_theta(opt.x)

        return final_theta

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
        assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]))
        return U
