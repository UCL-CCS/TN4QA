# from timeit import default_timer
# from typing import Callable

# from math import comb
# import numpy as np
# from qiskit import QuantumCircuit

# from ...dmrg import DMRG
# from ...mpo import MatrixProductOperator
# from ...mps import MatrixProductState
# from ..backend.base import QuantumBackend
# from ..backend.qiskit_simulator import QiskitSimulatorBackend
# from ..base import QuantumAlgorithm
# from ..result import Result
# from .controlled_time_evolved_qsci import ControlledTimeEvolvedQSCI
# from .time_evolved_qsci import TimeEvolvedQSCI


# class IterativeQSCI(QuantumAlgorithm):
#     def __init__(
#         self,
#         hamiltonian: dict,
#         niters: int,
#         method: str,
#         method_args: dict,
#         num_electrons: int,
#         dmrg_max_bond: int = 16,
#         dmrg_maxiter: int = 10,
#         scoring_function: Callable | None = None,
#         backend: QuantumBackend | None = None,
#         max_new_per_iter: int | None = None,
#         initial_state: MatrixProductState | str | None = None,
#     ) -> "IterativeQSCI":
#         """
#         Parameters
#         ----------
#         hamiltonian       : Pauli Hamiltonian dict {bitstring: coefficient}
#         niters            : number of QSCI iterations
#         method            : "TE" or "CTE"
#         method_args       : kwargs forwarded to the underlying QSCI class
#         num_electrons     : total number of electrons (sets HF reference)
#         dmrg_max_bond     : bond dimension for the initial DMRG reference state
#         dmrg_maxiter      : number of DMRG sweeps
#         scoring_function  : optional override for the reference-state scoring;
#                             signature f(iteration, amplitude, discovery_coeff) -> float
#         backend           : QuantumBackend instance (defaults to Qiskit simulator)
#         max_new_per_iter  : maximum number of *new* determinants that may be added
#                             to the accumulated important-configuration pool per
#                             iteration.  New determinants are ranked by |amplitude|^2
#                             and only the top-max_new_per_iter are kept.
#                             None (default) applies no per-iteration cap.

#                             This is independent of the global `subspace_size` passed
#                             to run(), which remains the cap on determinants sent to
#                             Davidson's diagonalisation each iteration.
#         """
#         self.method = method
#         if method in ("TE", "CTE"):
#             self.hamiltonian = self.sanitize_dict(hamiltonian)
#         else:
#             raise ValueError(f"Unknown method '{method}'. Choose 'TE' or 'CTE'.")

#         self.method_args = method_args
#         self.dmrg_max_bond = dmrg_max_bond
#         self.dmrg_maxiter = dmrg_maxiter
#         self.n_spinorbs = len(list(self.hamiltonian.keys())[0])
#         self.nelec = num_electrons
#         # self.initial_reference_state = self.run_dmrg()
#         if initial_state is None:
#             self.initial_reference_state = MatrixProductState.from_hf_state(self.n_spinorbs, self.nelec)
#         elif isinstance(initial_state, MatrixProductState):
#             self.initial_reference_state = initial_state
#         elif initial_state == "dicke":
#             self.initial_reference_state = self.dicke_state()
#         self.niters = niters
#         self.scoring_function = scoring_function
#         self.max_new_per_iter = max_new_per_iter
#         self.backend = self.set_backend(backend)

#         self.circuits = []

#         self.all_results = []
#         self.all_energies = []
#         self.all_subspace_sizes = []   # size of subspace AFTER Davidson each iteration
#         self.all_subspaces = []
#         self.all_groundstates = []
#         self.important_configurations = []
#         self.unimportant_configurations = []
#         self.all_circuit_depths = []
#         self.new_dets_per_iter = []    # how many new dets were admitted each iteration

#         self.cached_discover_coefficients = {}

#     @property
#     def circuit(self) -> QuantumCircuit:
#         return self.circuits

#     def sanitize_dict(self, d: dict[str, complex | float]) -> dict[str, float]:
#         return {
#             k: float(v.real) if isinstance(v, complex) else float(v)
#             for k, v in d.items()
#         }

#     def dicke_state(self) -> MatrixProductState:
#         """Build a Dicke state D(N,k) as an MPS

#         Args:
#             num_qubits: N = number of sites in MPS
#             num_ones: k = number of ones in each basis state
#             normalise: If set to False produces the unnormalised state (e.g., D(3,2) = |011> + |101> + |110>)

#         Returns:
#             An MPS representing D(N,k)
#         """
#         num_qubits = self.n_spinorbs
#         num_ones = self.nelec
#         normalise=True

#         D = num_ones + 1

#         # transition matrices
#         A0 = np.eye(D)
#         A1 = np.zeros((D, D))
#         for r in range(D - 1):
#             A1[r, r + 1] = 1.0

#         tensors = []

#         # First site: (D,2)
#         T0 = np.zeros((2, D))
#         T0[0,:] = A0[0,:]
#         T0[1,:] = A1[0,:]
#         T0 = np.moveaxis(T0, [0], [1])
#         tensors.append(T0)

#         # Bulk: (D,D,2)
#         for _ in range(1, num_qubits-1):
#             T = np.stack([A0, A1], axis=0)
#             T = np.moveaxis(T, [0], [2])
#             tensors.append(T)

#         # Last site: (D,2)
#         TN = np.zeros((2, D))
#         TN[0, :] = A0[:, num_ones]
#         TN[1, :] = A1[:, num_ones]
#         TN = np.moveaxis(TN, [0], [1])
#         tensors.append(TN)

#         if normalise:
#             norm = np.sqrt(comb(num_qubits, num_ones))
#             tensors = [T / (norm ** (1/num_qubits)) for T in tensors]

#         mps = MatrixProductState.from_arrays(tensors)

#         return mps

#     # def run_dmrg(self) -> MatrixProductState:
#     #     hf_state = MatrixProductState.from_hf_state(self.n_spinorbs, self.nelec)
#     #     dmrg = DMRG(
#     #         self.hamiltonian,
#     #         max_mps_bond=self.dmrg_max_bond,
#     #         initial_mps=hf_state,
#     #     )
#     #     _, gs = dmrg.run(nsweeps=self.dmrg_maxiter)
#     #     return gs

#     def run_one_shot(
#         self,
#         num_shots: int,
#         subspace_size: int,
#         reference_state: MatrixProductState,
#     ) -> tuple[list[str], np.ndarray]:
#         """
#         Run a single QSCI iteration.

#         Samples determinants, runs Davidson diagonalisation, and records
#         tracking data.  The subspace size stored in all_subspace_sizes is
#         the number of determinants that entered Davidson (i.e. post-sampling,
#         pre-Davidson), which is the correct measure of classical diagonalisation
#         cost.

#         Returns
#         -------
#         subspace : list of bitstrings that entered Davidson
#         gs       : ground-state coefficient vector from Davidson
#         """
#         cls = ControlledTimeEvolvedQSCI if self.method == "CTE" else TimeEvolvedQSCI
#         qsci = cls(
#             self.hamiltonian,
#             reference_state,
#             backend=self.backend,
#             known_important_configurations=self.important_configurations,
#             known_unimportant_configurations=self.unimportant_configurations,
#             **self.method_args,
#         )
#         self.circuits = qsci.circuit
#         result = qsci.run(num_shots, subspace_size)
#         self.all_results.append(result)

#         energy, gs = result.result
#         subspace = result.metadata["subspace"]

#         # Store subspace size as the number of determinants that reached Davidson.
#         # This is len(subspace) — the set actually diagonalised — rather than
#         # result.metadata["actual_subspace_size"], which is set before Davidson
#         # and may differ if the sampler deduplicates internally.
#         self.all_energies.append(energy)
#         self.all_groundstates.append(gs)
#         self.all_subspace_sizes.append(len(subspace))
#         self.all_subspaces.append(subspace)
#         self.all_circuit_depths.append(result.metadata["avg_circuit_depth"])

#         return subspace, gs

#     def _update_important_configurations(
#         self,
#         subspace: list[str],
#         gs: np.ndarray,
#         max_new: int | None,
#     ) -> int:
#         """
#         Add newly discovered determinants to the important-configuration pool.

#         Parameters
#         ----------
#         subspace : determinants from this iteration's Davidson step
#         gs       : corresponding ground-state amplitudes
#         max_new  : cap on new additions (None = unlimited)

#         Returns
#         -------
#         n_new : number of new determinants actually added
#         """
#         known_important = set(self.important_configurations)
#         known_unimportant = set(self.unimportant_configurations)

#         # Separate new determinants from already-known ones
#         candidates = [
#             (subspace[i], gs[i])
#             for i in range(len(subspace))
#             if subspace[i] not in known_important
#             and subspace[i] not in known_unimportant
#             and np.abs(gs[i]) ** 2 > 1e-16
#         ]

#         # Rank candidates by amplitude magnitude (highest first)
#         candidates.sort(key=lambda x: np.abs(x[1]) ** 2, reverse=True)

#         if max_new is not None:
#             candidates = candidates[:max_new]

#         new_bitstrings = [c[0] for c in candidates]
#         self.important_configurations = list(
#             known_important | set(new_bitstrings)
#         )

#         # Anything sampled but not admitted as important goes to unimportant
#         admitted = set(new_bitstrings)
#         for i in range(len(subspace)):
#             bs = subspace[i]
#             if (
#                 bs not in known_important
#                 and bs not in admitted
#                 and np.abs(gs[i]) ** 2 <= 1e-16
#             ):
#                 self.unimportant_configurations.append(bs)
#         self.unimportant_configurations = list(set(self.unimportant_configurations))

#         return len(new_bitstrings)

#     def calculate_discovery_coefficient(self, bitstring: MatrixProductState) -> float:
#         if isinstance(self.scoring_function, Callable):
#             if self.scoring_function.__name__ == "exploitation_scoring":
#                 return 0.0
#         if bitstring in self.cached_discover_coefficients:
#             return self.cached_discover_coefficients[bitstring]
#         ham_mpo = MatrixProductOperator.from_hamiltonian(self.hamiltonian)
#         h_bitstring = bitstring.apply_mpo(ham_mpo)
#         ip = h_bitstring.compute_inner_product(h_bitstring).real
#         exp_val = np.abs(h_bitstring.compute_inner_product(bitstring)) ** 2
#         diff = max(ip - exp_val, 0)
#         dc = np.sqrt(diff)
#         self.cached_discover_coefficients[bitstring] = dc
#         return dc

#     def calculate_scoring_function(
#         self,
#         iteration_number: int,
#         amplitude: float,
#         discovery_coefficient: float,
#     ) -> float:
#         if self.scoring_function is None:
#             lam = iteration_number / self.niters
#             return lam * amplitude + (1 - lam) * discovery_coefficient
#         return self.scoring_function(iteration_number, amplitude, discovery_coefficient)

#     def prepare_reference_state(
#         self,
#         iteration_number: int,
#         subspace: list[str],
#         gs: np.ndarray,
#     ) -> MatrixProductState:
#         discovery_coeffs = []
#         for idx in range(len(subspace)):
#             bitstring = MatrixProductState.from_bitstring(subspace[idx])
#             d = self.calculate_discovery_coefficient(bitstring)
#             discovery_coeffs.append(d)

#         total_d = sum(np.abs(d) ** 2 for d in discovery_coeffs)
#         if total_d != 0:
#             normalised_discovery_coeffs = [d / np.sqrt(total_d) for d in discovery_coeffs]
#         else:
#             normalised_discovery_coeffs = discovery_coeffs

#         scores = [
#             self.calculate_scoring_function(
#                 iteration_number, gs[idx], normalised_discovery_coeffs[idx]
#             )
#             for idx in range(len(subspace))
#         ]

#         total_s = sum(f ** 2 for f in scores)
#         if total_s != 0:
#             weights = [f / np.sqrt(total_s) for f in scores]
#         else:
#             weights = [1 / np.sqrt(len(subspace)) for _ in scores]

#         d = {subspace[idx]: weights[idx] for idx in range(len(subspace))}
#         mps = MatrixProductState.from_bitstring_dict(d)
#         mps.compress(2)
#         return mps

#     def run(self, num_shots: int, subspace_size: int) -> Result:
#         """
#         Run the iterative QSCI algorithm.

#         Parameters
#         ----------
#         num_shots    : total shot budget, split equally across iterations
#         subspace_size: maximum number of determinants passed to Davidson each
#                        iteration (the global cap).  Use max_new_per_iter in
#                        __init__ to additionally cap how many *new* determinants
#                        are admitted to the accumulated pool per iteration.
#         """
#         start_timer = default_timer()
#         reference_state = self.initial_reference_state
#         shots_per_iteration = int(num_shots / self.niters)

#         for iteration in range(self.niters):
#             print(f"Iteration {iteration + 1}/{self.niters}")

#             max_subspace = self.max_new_per_iter * (iteration+1)
#             subspace, gs = self.run_one_shot(
#                 shots_per_iteration, max_subspace, reference_state
#             )

#             n_new = self._update_important_configurations(
#                 subspace, gs, self.max_new_per_iter
#             )
#             self.new_dets_per_iter.append(n_new)
#             print(
#                 f"  subspace size (post-Davidson): {len(subspace)}"
#                 f"  |  new dets admitted: {n_new}"
#                 f"  |  pool size: {len(self.important_configurations)}"
#             )

#             if len(self.important_configurations) > 0:
#                 reference_state = self.prepare_reference_state(
#                     iteration, self.important_configurations, gs
#                 )

#         end_timer = default_timer()

#         metadata = {
#             "algorithm_name": "IterativeQSCI",
#             "num_shots": num_shots,
#             "max_subspace_size": subspace_size,
#             "max_new_per_iter": self.max_new_per_iter,
#             "all_energies": self.all_energies,
#             "all_subspaces": self.all_subspaces,
#             "all_subspace_sizes": self.all_subspace_sizes,
#             "all_groundstates": self.all_groundstates,
#             "all_circuit_depths": self.all_circuit_depths,
#             "new_dets_per_iter": self.new_dets_per_iter,
#             "total_runtime": end_timer - start_timer,
#         }
#         if self.backend is not None:
#             metadata["backend_name"] = self.backend.name
#             metadata["backend_coupling_map"] = self.backend.coupling_map
#             metadata["backend_basis_gates"] = self.backend.basis_gates
#             metadata["backend_num_qubits"] = self.backend.num_qubits

#         result = Result(
#             result=(self.all_energies[-1], self.all_groundstates[-1]),
#             measurements=None,
#             parameters=None,
#             metadata=metadata,
#         )
#         return result

#     def set_backend(self, backend: QuantumBackend | None) -> None:
#         if backend is None:
#             backend = QiskitSimulatorBackend()
#         self.backend = backend
#         return

from math import comb
from timeit import default_timer
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit

from ...mps import MatrixProductState
from ..backend.base import QuantumBackend
from ..backend.qiskit_simulator import QiskitSimulatorBackend
from ..base import QuantumAlgorithm
from ..result import Result
from .controlled_time_evolved_qsci import ControlledTimeEvolvedQSCI
from .time_evolved_qsci import TimeEvolvedQSCI

# ---------------------------------------------------------------------------
# Module-level helpers for fast discovery coefficient computation
# ---------------------------------------------------------------------------


def _popcount_array(x: np.ndarray) -> np.ndarray:
    """
    Compute popcount (number of set bits) for every element of a uint64 array.
    Uses the Hamming-weight bit-twiddling identity — faster than a Python loop
    and works on any numpy version.
    """
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)


def _build_sparse_ham(hamiltonian: dict):
    """
    Pre-process a Pauli Hamiltonian into integer masks for fast bitwise
    action on computational basis states.

    For a Pauli string P = P_0 P_1 ... P_{n-1} acting on |j⟩:
      * Bits at X/Y positions are flipped  →  flip_mask (XOR with input integer)
      * Z/Y positions contribute a (-1) factor per set bit in the input
        →  z_mask  (AND with input, then count bits mod 2 for sign)
      * Each Y position contributes an extra factor of i (independent of input)
        →  accumulated into y_base_phase = i^(number of Y sites)

    Parameters
    ----------
    hamiltonian : dict {pauli_string: coefficient}

    Returns
    -------
    flip_masks     : (L,) int64 — XOR mask to get output bitstring
    z_masks        : (L,) int64 — AND mask for Z/Y phase parity
    y_base_phases  : (L,) complex — i^(n_Y) factor per term
    coeffs         : (L,) complex — Hamiltonian coefficients
    n_qubits       : int
    """
    keys = list(hamiltonian.keys())
    n = len(keys[0])
    assert n <= 62, "Bitwise discovery coefficients require n_qubits <= 62"
    L = len(keys)

    coeffs = np.array([hamiltonian[k] for k in keys], dtype=complex)
    flip_masks = np.zeros(L, dtype=np.int64)
    z_masks = np.zeros(L, dtype=np.int64)
    y_count = np.zeros(L, dtype=np.int32)

    for l, key in enumerate(keys):
        fm = 0
        zm = 0
        yc = 0
        for i, c in enumerate(key):
            bit = n - 1 - i  # MSB-first ordering
            if c == "X":
                fm |= 1 << bit
            elif c == "Y":
                fm |= 1 << bit
                zm |= 1 << bit
                yc += 1
            elif c == "Z":
                zm |= 1 << bit
        flip_masks[l] = fm
        z_masks[l] = zm
        y_count[l] = yc

    i_powers = np.array([1, 1j, -1, -1j], dtype=complex)
    y_base_phases = i_powers[y_count % 4]

    return flip_masks, z_masks, y_base_phases, coeffs, n


def _batch_discovery_coefficients(
    input_ints: np.ndarray,
    flip_masks: np.ndarray,
    z_masks: np.ndarray,
    y_base_phases: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Compute discovery coefficients d_j = ||H_off |j>||_2 for a batch of
    computational basis states using integer bitwise operations.

    d_j^2 = ||H_off |j>||^2
          = sum_{k != j} |sum_{l: P_l|j> = |k>} c_l * phase_l(j)|^2

    This replaces the TN contraction approach.  Complexity is O(L * B) where
    L = number of Hamiltonian terms and B = number of bitstrings, vs the TN
    approach which scales exponentially in bond dimension.

    Parameters
    ----------
    input_ints    : (B,) int64 — each bitstring encoded as an integer
    flip_masks    : (L,) int64 from _build_sparse_ham
    z_masks       : (L,) int64 from _build_sparse_ham
    y_base_phases : (L,) complex from _build_sparse_ham
    coeffs        : (L,) complex from _build_sparse_ham

    Returns
    -------
    (B,) float64 — discovery coefficient for each bitstring
    """
    B = len(input_ints)

    # Output integers: input XOR flip_mask, broadcast over (L, B)
    output_ints = input_ints[None, :] ^ flip_masks[:, None]  # (L, B) int64

    # Phase from Z/Y sites: (-1)^popcount(input & z_mask)
    zy_bits = input_ints[None, :] & z_masks[:, None]  # (L, B) int64
    parities = _popcount_array(zy_bits.astype(np.uint64)) % 2  # (L, B) int32
    z_phases = np.where(parities == 1, -1.0 + 0j, 1.0 + 0j)  # (L, B) complex

    # Total weighted amplitude for each (term, bitstring) pair
    total_phases = y_base_phases[:, None] * z_phases  # (L, B) complex
    weighted = coeffs[:, None] * total_phases  # (L, B) complex

    # Accumulate by output state and compute off-diagonal norm
    results = np.empty(B)
    for b in range(B):
        out = output_ints[:, b]  # (L,) int64
        w = weighted[:, b]  # (L,) complex
        unique_out, inv = np.unique(out, return_inverse=True)
        acc = np.zeros(len(unique_out), dtype=complex)
        np.add.at(acc, inv, w)
        off_diag_mask = unique_out != input_ints[b]
        results[b] = np.sqrt(
            max(float(np.sum(np.abs(acc[off_diag_mask]) ** 2).real), 0.0)
        )

    return results


class IterativeQSCI(QuantumAlgorithm):
    def __init__(
        self,
        hamiltonian: dict,
        niters: int,
        method: str,
        method_args: dict,
        num_electrons: int,
        dmrg_max_bond: int = 16,
        dmrg_maxiter: int = 10,
        scoring_function: Callable | None = None,
        backend: QuantumBackend | None = None,
        max_new_per_iter: int | None = None,
        initial_state: "MatrixProductState | str | None" = None,
    ) -> "IterativeQSCI":
        """
        Parameters
        ----------
        hamiltonian       : Pauli Hamiltonian dict {bitstring: coefficient}
        niters            : number of QSCI iterations
        method            : "TE" or "CTE"
        method_args       : kwargs forwarded to the underlying QSCI class
        num_electrons     : total number of electrons (sets HF reference)
        dmrg_max_bond     : bond dimension for the initial DMRG reference state
        dmrg_maxiter      : number of DMRG sweeps
        scoring_function  : optional override for the reference-state scoring;
                            signature f(iteration, amplitude, discovery_coeff) -> float
        backend           : QuantumBackend instance (defaults to Qiskit simulator)
        max_new_per_iter  : maximum number of *new* determinants that may be added
                            to the accumulated important-configuration pool per
                            iteration.  New determinants are ranked by |amplitude|^2
                            and only the top-max_new_per_iter are kept.
                            None (default) applies no per-iteration cap.
        initial_state     : initial reference state.  One of:
                              None         — Hartree-Fock bitstring state
                              "dicke"      — equal superposition over all n-electron dets
                              MatrixProductState — use directly
        """
        self.method = method
        if method in ("TE", "CTE"):
            self.hamiltonian = self.sanitize_dict(hamiltonian)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'TE' or 'CTE'.")

        self.method_args = method_args
        self.dmrg_max_bond = dmrg_max_bond
        self.dmrg_maxiter = dmrg_maxiter
        self.n_spinorbs = len(list(self.hamiltonian.keys())[0])
        self.nelec = num_electrons

        if initial_state is None:
            self.initial_reference_state = MatrixProductState.from_hf_state(
                self.n_spinorbs, self.nelec
            )
        elif isinstance(initial_state, MatrixProductState):
            self.initial_reference_state = initial_state
        elif initial_state == "dicke":
            self.initial_reference_state = self._dicke_state()
        else:
            raise ValueError(f"Unknown initial_state '{initial_state}'.")

        self.niters = niters
        self.scoring_function = scoring_function
        self.max_new_per_iter = max_new_per_iter
        self.backend = self.set_backend(backend)

        # Pre-process Hamiltonian into integer masks once at construction time.
        # These are reused for every discovery coefficient calculation.
        (
            self._flip_masks,
            self._z_masks,
            self._y_base_phases,
            self._coeffs_complex,
            self._n_qubits,
        ) = _build_sparse_ham(self.hamiltonian)

        self.circuits = []

        self.all_results = []
        self.all_energies = []
        self.all_subspace_sizes = []
        self.all_subspaces = []
        self.all_groundstates = []
        self.important_configurations = []
        self.unimportant_configurations = []
        self.all_circuit_depths = []
        self.new_dets_per_iter = []

    @property
    def circuit(self) -> QuantumCircuit:
        return self.circuits

    def sanitize_dict(self, d: dict) -> dict:
        return {
            k: float(v.real) if isinstance(v, complex) else float(v)
            for k, v in d.items()
        }

    def _dicke_state(self) -> MatrixProductState:
        """Equal superposition over all n-electron computational basis states."""
        N = self.n_spinorbs
        k = self.nelec
        D = k + 1

        A0 = np.eye(D)
        A1 = np.zeros((D, D))
        for r in range(D - 1):
            A1[r, r + 1] = 1.0

        tensors = []
        T0 = np.zeros((2, D))
        T0[0] = A0[0]
        T0[1] = A1[0]
        tensors.append(np.moveaxis(T0, 0, 1))

        for _ in range(1, N - 1):
            T = np.stack([A0, A1], axis=0)
            tensors.append(np.moveaxis(T, 0, 2))

        TN = np.zeros((2, D))
        TN[0] = A0[:, k]
        TN[1] = A1[:, k]
        tensors.append(np.moveaxis(TN, 0, 1))

        norm = np.sqrt(comb(N, k))
        tensors = [T / (norm ** (1.0 / N)) for T in tensors]
        return MatrixProductState.from_arrays(tensors)

    # ------------------------------------------------------------------ #
    # Discovery coefficients — fast bitwise implementation                #
    # ------------------------------------------------------------------ #

    def calculate_discovery_coefficients(self, bitstrings: list[str]) -> np.ndarray:
        """
        Compute d_j = ||H_off |j>||_2 for a batch of bitstrings.

        Replaces the old TN-contraction approach with direct integer bitwise
        arithmetic.  Complexity O(L * B) vs the old O(L * chi^n).

        Parameters
        ----------
        bitstrings : list of bitstring strings, e.g. ["1100", "1010"]

        Returns
        -------
        np.ndarray of shape (B,) with discovery coefficient per bitstring
        """
        input_ints = np.array([int(bs, 2) for bs in bitstrings], dtype=np.int64)
        return _batch_discovery_coefficients(
            input_ints,
            self._flip_masks,
            self._z_masks,
            self._y_base_phases,
            self._coeffs_complex,
        )

    # kept for backwards-compatibility but now delegates to the batch method
    def calculate_discovery_coefficient(self, bitstring) -> float:
        """Single-bitstring wrapper around calculate_discovery_coefficients."""
        bs = bitstring if isinstance(bitstring, str) else bitstring.to_bitstring()
        return float(self.calculate_discovery_coefficients([bs])[0])

    # ------------------------------------------------------------------ #

    def run_one_shot(
        self,
        num_shots: int,
        subspace_size: int,
        reference_state: MatrixProductState,
    ) -> tuple[list[str], np.ndarray]:
        cls = ControlledTimeEvolvedQSCI if self.method == "CTE" else TimeEvolvedQSCI
        qsci = cls(
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

        self.all_energies.append(energy)
        self.all_groundstates.append(gs)
        self.all_subspace_sizes.append(len(subspace))
        self.all_subspaces.append(subspace)
        self.all_circuit_depths.append(result.metadata["avg_circuit_depth"])

        return subspace, gs

    def _update_important_configurations(
        self,
        subspace: list[str],
        gs: np.ndarray,
        max_new: int | None,
    ) -> int:
        known_important = set(self.important_configurations)
        known_unimportant = set(self.unimportant_configurations)

        candidates = [
            (subspace[i], gs[i])
            for i in range(len(subspace))
            if subspace[i] not in known_important
            and subspace[i] not in known_unimportant
            and np.abs(gs[i]) ** 2 > 1e-16
        ]
        candidates.sort(key=lambda x: np.abs(x[1]) ** 2, reverse=True)
        if max_new is not None:
            candidates = candidates[:max_new]

        new_bitstrings = [c[0] for c in candidates]
        self.important_configurations = list(known_important | set(new_bitstrings))

        admitted = set(new_bitstrings)
        for i in range(len(subspace)):
            bs = subspace[i]
            if (
                bs not in known_important
                and bs not in admitted
                and np.abs(gs[i]) ** 2 <= 1e-16
            ):
                self.unimportant_configurations.append(bs)
        self.unimportant_configurations = list(set(self.unimportant_configurations))

        return len(new_bitstrings)

    def calculate_scoring_function(
        self,
        iteration_number: int,
        amplitude: float,
        discovery_coefficient: float,
    ) -> float:
        if self.scoring_function is None:
            lam = iteration_number / self.niters
            return lam * amplitude + (1 - lam) * discovery_coefficient
        return self.scoring_function(iteration_number, amplitude, discovery_coefficient)

    def prepare_reference_state(
        self,
        iteration_number: int,
        subspace: list[str],
        gs: np.ndarray,
    ) -> MatrixProductState:
        # Compute all discovery coefficients in one batch call
        is_exploit = (
            self.scoring_function is not None
            and getattr(self.scoring_function, "__name__", "") == "exploitation_scoring"
        )
        if is_exploit:
            discovery_coeffs = np.zeros(len(subspace))
        else:
            discovery_coeffs = self.calculate_discovery_coefficients(subspace)

        total_d = np.sum(discovery_coeffs**2)
        if total_d > 0:
            normalised_dc = discovery_coeffs / np.sqrt(total_d)
        else:
            normalised_dc = discovery_coeffs

        scores = np.array(
            [
                self.calculate_scoring_function(
                    iteration_number, gs[i], normalised_dc[i]
                )
                for i in range(len(subspace))
            ]
        )

        total_s = np.sum(scores**2)
        if total_s > 0:
            weights = scores / np.sqrt(total_s)
        else:
            weights = np.full(len(subspace), 1.0 / np.sqrt(len(subspace)))

        d = {subspace[i]: float(weights[i]) for i in range(len(subspace))}
        mps = MatrixProductState.from_bitstring_dict(d)
        mps.compress(2)
        return mps

    def run(self, num_shots: int, subspace_size: int) -> Result:
        """
        Run the iterative QSCI algorithm.

        Parameters
        ----------
        num_shots    : total shot budget, split equally across iterations
        subspace_size: maximum number of determinants passed to Davidson each
                       iteration (the global cap).
        """
        start_timer = default_timer()
        reference_state = self.initial_reference_state
        shots_per_iteration = int(num_shots / self.niters)

        for iteration in range(self.niters):
            print(f"Iteration {iteration + 1}/{self.niters}")

            max_subspace = self.max_new_per_iter * (iteration + 1)
            subspace, gs = self.run_one_shot(
                shots_per_iteration, max_subspace, reference_state
            )

            n_new = self._update_important_configurations(
                subspace, gs, self.max_new_per_iter
            )
            self.new_dets_per_iter.append(n_new)
            print(
                f"  subspace size (post-Davidson): {len(subspace)}"
                f"  |  new dets admitted: {n_new}"
                f"  |  pool size: {len(self.important_configurations)}"
            )

            if len(self.important_configurations) > 0 and iteration < self.niters - 1:
                reference_state = self.prepare_reference_state(
                    iteration, self.important_configurations, gs
                )

        end_timer = default_timer()

        metadata = {
            "algorithm_name": "IterativeQSCI",
            "num_shots": num_shots,
            "max_subspace_size": subspace_size,
            "max_new_per_iter": self.max_new_per_iter,
            "all_energies": self.all_energies,
            "all_subspaces": self.all_subspaces,
            "all_subspace_sizes": self.all_subspace_sizes,
            "all_groundstates": self.all_groundstates,
            "all_circuit_depths": self.all_circuit_depths,
            "new_dets_per_iter": self.new_dets_per_iter,
            "total_runtime": end_timer - start_timer,
        }
        if self.backend is not None:
            metadata["backend_name"] = self.backend.name
            metadata["backend_coupling_map"] = self.backend.coupling_map
            metadata["backend_basis_gates"] = self.backend.basis_gates
            metadata["backend_num_qubits"] = self.backend.num_qubits

        return Result(
            result=(self.all_energies[-1], self.all_groundstates[-1]),
            measurements=None,
            parameters=None,
            metadata=metadata,
        )

    def set_backend(self, backend: QuantumBackend | None) -> None:
        if backend is None:
            backend = QiskitSimulatorBackend()
        self.backend = backend
        return
