from typing import Callable

import numpy as np
from numpy import ndarray
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner
from scipy.optimize import minimize
from symmer.operators import PauliwordOp

from tn4qa.qi_cost_functions import (
    cost_function_dict_to_purity_mpo,
    cost_function_to_dict,
)

from ..dmrg import DMRG
from ..mpo import MatrixProductOperator
from ..mps import MatrixProductState
from ..quantum_algorithms.hamiltonian_simulation.trotterisation import TrotterSimulation


class ActiveSpaceSelection:
    def __init__(self, hamiltonian: dict[str, complex], coeff_matrix: ndarray):
        """Constructor

        Args:
            hamiltonian: System Hamiltonian
            coeff_matrix: HF coefficient matrix of shape (N, N)
        """
        self.hamiltonian = hamiltonian
        self.num_spin_orbitals = coeff_matrix.shape[0]
        self.num_orbitals = int(self.num_spin_orbitals / 2)
        self.coeff_matrix = coeff_matrix

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

        Returns:
            Transformed coefficient matrix with optimal active orbitals
        """
        function_args = kwargs
        N = self.num_spin_orbitals
        assert self.coeff_matrix.shape[1] == N, "Coefficient matrix must be square"

        # Write the Hamiltonian and perfrom DMRG to get the initial state |psi>_C
        max_mps_bond = function_args.get("dmrg_max_mps_bond", 8)
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

        # Cost function to MPO
        decay_power = function_args.get("cost_function_decay_power", 2.0)
        cost_mpo = self.build_cost_function_mpo(
            cost_function=cost_function, decay_power=decay_power
        )

        # Run BFGS optimisation to find optimal K
        theta_init = np.zeros((N**2,), dtype=float)  # Initial guess for theta
        theta_opt = self.optimise_K(theta_init, cost_mpo, psi_C)

        # Exponentiate K to get a unitary U = exp(K)
        K_opt = self.vector_to_antihermitian(theta_opt)
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
        self, cost_function: Callable, decay_power: float
    ) -> MatrixProductOperator:
        """Build the cost function as an MPO"""
        d = cost_function_to_dict(
            cost_function, num_orbitals=self.num_orbitals, decay_power=decay_power
        )
        mpo = cost_function_dict_to_purity_mpo(self.num_spin_orbitals, d)
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

    def exponential_hopping_term(
        self, p: int, q: int, K_pq: complex
    ) -> MatrixProductOperator:
        """
        Construct the MPO for exp(K_pq * a_p† a_q - K_pq* * a_q† a_p).

        Args:
            p, q: Indices of orbitals (must be different)
            K_pq: Complex parameter

        Returns:
            MatrixProductOperator for exp(H), where H = K_pq * a_p† a_q - conj(K_pq) * a_q† a_p
        """
        if p == q:
            h_fermion = FermionOperator(((p, 1), (q, 0)), K_pq)

        else:
            # Build the FermionOperator, 1 is the creation operator, 0 is the annihilation operator
            # H = K_pq * a_p† a_q - conj(K_pq) * a_q† a_p
            h_fermion = FermionOperator(((p, 1), (q, 0)), K_pq) - FermionOperator(
                ((q, 1), (p, 0)), np.conj(K_pq)
            )

        # Map to QubitOperator using Jordan-Wigner
        h_qubit = jordan_wigner(h_fermion)

        # Convert to PauliwordOp
        h_pauli = PauliwordOp.from_openfermion(h_qubit, n_qubits=self.num_spin_orbitals)

        # Convert to a dictionary
        h_dict = h_pauli.to_dictionary
        h_dict = {k: v.real for k, v in h_dict.items()}

        # Create a circuit
        sim = TrotterSimulation(h_dict, duration=1.0, num_steps=1)
        qc = sim.circuit

        # Convert Qiskit circuit to MPO
        u_mpo = MatrixProductOperator.from_qiskit_circuit(qc)

        return u_mpo

    def build_trotterised_unitary(
        self, K: ndarray, trotter_steps: int = 1
    ) -> MatrixProductOperator:
        """
        Build an MPO approximation of the fermionic unitary:
            U = exp(Σ_{pq} K_{pq} a†_p a_q)

        using first-order Trotter decomposition.

        Args:
            K: Anti-Hermitian matrix (N x N)
            trotter_steps: Number of Trotter steps

        Returns:
            MatrixProductOperator representing the unitary
        """
        N = K.shape[0]
        assert K.shape[1] == N, "K must be square"
        assert np.allclose(K + K.conj().T, 0, atol=1e-10), "K must be anti-Hermitian"

        u_mpo = MatrixProductOperator.identity_mpo(N)
        dt = 1.0 / trotter_steps

        for _ in range(trotter_steps):
            for p in range(N):
                for q in range(p, N):
                    if abs(K[p, q]) > 1e-12:
                        K_dt = dt * K[p, q]
                        hop_exp_mpo = self.exponential_hopping_term(p, q, K_dt)
                        u_mpo = u_mpo * hop_exp_mpo

        return u_mpo

    def optimisation_cost(
        self, theta: ndarray, mpo: MatrixProductOperator, mps: MatrixProductState
    ) -> float:
        """Optimisation cost function.

        Args;
            theta: Paramter list for K
            mpo: QI cost function MPO
            mps: Groundstate approximation from DMRG

        Returns:
            < MPS | (exp(Σ_{pq} K_{pq} a†_p a_q))† MPO exp(Σ_{pq} K_{pq} a†_p a_q) | MPS >
        """
        # N = V.num_sites
        K = self.vector_to_antihermitian(theta)
        W_rotated = self.build_trotterised_unitary(K)  # returns an MPO
        transformed_state = mps.apply_mpo(W_rotated)
        doubled_transformed_state = transformed_state.to_two_copy_mps()
        cost = doubled_transformed_state.compute_expectation_value(mpo)
        return cost.real

    def optimise_K(
        self, theta_init: ndarray, mpo: MatrixProductOperator, mps: MatrixProductState
    ):
        """
        Run BFGS optimisation over K to minimise optimisation_cost.

        Args:
            theta_init: Initial guess for theta
            mpo: QI cost function MPO
            mps: Groundstate approximation from DMRG

        Returns:
            Optimal real-valued parameter vector θ defining anti-Hermitian K
        """
        # N = V.num_sites
        # num_params = N**2
        # theta0 = np.zeros(num_params)

        result = minimize(
            self.optimisation_cost,
            theta_init,
            args=(mpo, mps),
            method="COBYLA",
            options={"disp": True},
        )
        print("Optimisation result:", result.x)
        return result.x

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
