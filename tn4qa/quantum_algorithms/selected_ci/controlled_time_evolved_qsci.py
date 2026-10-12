# import copy
# from typing import Callable

# import numpy as np
# from qiskit import QuantumCircuit

# from ...mps import MatrixProductState
# from ...quantum_algorithms.hamiltonian_simulation.qdrift import (
#     QDriftSimulation,
# )
# from ...quantum_algorithms.hamiltonian_simulation.trotterisation import (
#     TrotterSimulation,
# )
# from ...tn_methods.mps_to_circuit import MPSAnalyticDecomposition
# from ..backend.base import QuantumBackend
# from ..backend.qiskit_simulator import QiskitSimulatorBackend
# from ..base import QuantumAlgorithm
# from ..result import Result
# from ..utils import add_controls
# from .qsci import QSCI


# class ControlledTimeEvolvedQSCI(QuantumAlgorithm):
#     def __init__(
#         self,
#         hamiltonian: dict,
#         reference_state: MatrixProductState,
#         duration: float = np.pi,
#         num_circuits: int = 5,
#         backend: QuantumBackend | None = None,
#         qdrift: bool = True,
#         num_qdrift_circuits: int | None = 10,
#         qdrift_error: float | None = None,
#         max_qdrift_terms: int | None = None,
#         num_electrons: int | None = None,
#         known_important_configurations: list[str] = [],
#         known_unimportant_configurations: list[str] = [],
#         postprocessing_function: Callable | None = None,
#         postprocessing_args: dict | None = None,
#         alpha: float=0.5,
#     ) -> "ControlledTimeEvolvedQSCI":
#         """
#         Constructor for TE-QSCI class.
#         """
#         self.alpha = alpha
#         # self.duration = duration
#         self.circuits = []
#         self.num_circuits = num_circuits
#         self.norm = None
#         hamiltonian = self.sanitize_dict(hamiltonian)
#         hamiltonian = self.normalise_hamiltonian(hamiltonian)
#         self.hamiltonian = self.rescale_hamiltonian(hamiltonian)
#         # self.duration = (2 * self.norm * duration) / np.pi
#         self.duration = 2.0 * alpha
#         self.reference_state = reference_state
#         self.reference_state_qc = self.create_reference_circuit()
#         self.backend = self.set_backend(backend=backend)

#         self.qdrift = qdrift
#         self.num_qdrift_circuits = num_qdrift_circuits
#         self.qdrift_error = qdrift_error
#         self.max_qdrift_terms = max_qdrift_terms

#         self.num_electrons = num_electrons
#         self.important_configurations = known_important_configurations
#         self.unimportant_configurations = known_unimportant_configurations
#         self.postprocessing_function = postprocessing_function
#         self.postprocessing_args = postprocessing_args

#         self.circuit_depths = []
#         self.energy = None

#     @property
#     def circuit(self) -> QuantumCircuit:
#         """Get the circuit from the algorithm"""
#         return self.circuits

#     def sanitize_dict(self, d: dict[str, complex | float]) -> dict[str, float]:
#         return {
#             k: float(v.real) if isinstance(v, complex) else float(v)
#             for k, v in d.items()
#         }

#     def normalise_hamiltonian(self, d: dict[str, float]) -> dict[str, float]:
#         self.norm = np.sum([np.abs(x) for x in d.values()])
#         return {k: v / self.norm for k, v in d.items()}

#     def rescale_hamiltonian(self, d: dict[str, float]) -> dict[str, float]:
#         num_qubits = len(list(d.keys())[0])
#         d["I" * num_qubits] = d.get("I" * num_qubits, 0) + 1.0
#         return {k: v * 0.5 * np.pi / 2 for k, v in d.items()}

#     def create_reference_circuit(self) -> QuantumCircuit:
#         """Create a circuit to prepare the reference state"""
#         if self.reference_state.bond_dimension == 1:
#             mpstocirc = MPSAnalyticDecomposition(self.reference_state, 1, 1.0)
#             qc = mpstocirc.bond_dim_2_to_qc_exact(self.reference_state)
#         else:
#             self.reference_state.compress(2)
#             mpstocirc = MPSAnalyticDecomposition(self.reference_state, 1, 1.0)
#             qc = mpstocirc.mps_to_qc_via_ttn(self.reference_state, 2)
#         return qc

#     def perform_controlled_time_evolution(self, duration: float) -> QuantumCircuit:
#         """Add time evolution to the circuit"""
#         if duration == 0.0:
#             ref = copy.deepcopy(self.reference_state_qc)
#             return ref
#         sim = TrotterSimulation(self.hamiltonian, duration=duration)
#         sim_circ = sim.circuit
#         controlled_sim_circ = QuantumCircuit(sim_circ.num_qubits + 1)
#         controlled_sim_circ.compose(
#             sim_circ, qubits=range(1, sim_circ.num_qubits + 1), inplace=True
#         )
#         controlled_sim_circ = add_controls(controlled_sim_circ, [0])
#         circ_copy = copy.deepcopy(self.reference_state_qc)
#         ref = QuantumCircuit(circ_copy.num_qubits + 1)
#         ref.h(0)
#         ref.compose(circ_copy, qubits=range(1, circ_copy.num_qubits + 1), inplace=True)
#         ref.compose(controlled_sim_circ, inplace=True)
#         ref.h(0)
#         return ref

#     def perform_controlled_time_evolution_qdrift(
#         self, duration: float, error: float | None = None
#     ) -> QuantumCircuit:
#         """Add qdrift time evolution to the circuit"""
#         if duration == 0.0:
#             ref = copy.deepcopy(self.reference_state_qc)
#             return ref
#         sim = QDriftSimulation(
#             self.hamiltonian,
#             duration=duration,
#             error=error,
#             max_num_terms=self.max_qdrift_terms,
#         )
#         sim_circ = sim.circuit
#         controlled_sim_circ = QuantumCircuit(sim_circ.num_qubits + 1)
#         controlled_sim_circ.compose(
#             sim_circ, qubits=range(1, sim_circ.num_qubits + 1), inplace=True
#         )
#         controlled_sim_circ = add_controls(controlled_sim_circ, [0])
#         circ_copy = copy.deepcopy(self.reference_state_qc)
#         ref = QuantumCircuit(circ_copy.num_qubits + 1)
#         ref.h(0)
#         ref.compose(circ_copy, qubits=range(1, circ_copy.num_qubits + 1), inplace=True)
#         ref.compose(controlled_sim_circ, inplace=True)
#         ref.h(0)
#         return ref

#     def build_circuits_trotter(self, duration: float) -> list[QuantumCircuit]:
#         """Get circuits using Trotterisation"""
#         duration_per_circuit = duration / self.num_circuits
#         circuits = []
#         for idx in range(self.num_circuits):
#             qc = self.perform_controlled_time_evolution(
#                 (idx + 1) * duration_per_circuit
#             )
#             circuits.append(qc)
#         return circuits

#     def build_circuits_qdrift(self, duration: float) -> list[QuantumCircuit]:
#         """Get circuits using qDRIFT"""
#         duration_per_circuit = duration / (self.num_circuits)
#         circuits = []
#         for idx in range(self.num_circuits):
#             for _ in range(self.num_qdrift_circuits):
#                 qc = self.perform_controlled_time_evolution_qdrift(
#                     (idx + 1) * duration_per_circuit, error=self.qdrift_error
#                 )
#                 circuits.append(qc)
#         return circuits

#     def build_circuits(self) -> list[QuantumCircuit]:
#         if self.qdrift:
#             circuits = self.build_circuits_qdrift(self.duration)
#         else:
#             circuits = self.build_circuits_trotter(self.duration)
#         self.circuits = circuits
#         return circuits

#     def post_selection(self, counts: dict[str, int]) -> dict[str, int]:
#         """Post select counts based on the ancilla output"""
#         new_counts = {}
#         for b, count in counts.items():
#             if len(b) == self.reference_state.num_sites:
#                 new_counts[b] = new_counts.get(b, 0) + count
#             elif b[0] == "0":
#                 new_counts[b[1:]] = new_counts.get(b[1:], 0) + count
#         return new_counts

#     def rescale_energy(self, energy_prime):
#         return self.norm * ((2 / (0.5 * np.pi)) * energy_prime - 1)

#     def run(self, num_shots: int, subspace_size: int) -> Result:
#         """Run the full algorithm pipeline. Returns result object or final value."""
#         circuits = self.build_circuits()
#         if self.postprocessing_function is not None:
#             postprocessing_funcs = [self.post_selection] + self.postprocessing_function
#             func_args = [None] + self.postprocessing_args
#         else:
#             postprocessing_funcs = [self.post_selection]
#             func_args = [None]
#         qsci = QSCI(
#             circuits,
#             self.backend,
#             self.hamiltonian,
#             self.num_electrons,
#             self.important_configurations,
#             self.unimportant_configurations,
#             postprocessing_funcs,
#             func_args,
#         )
#         result = qsci.run(num_shots, subspace_size)
#         rescaled_energy = self.rescale_energy(result.result[0])
#         result.result = rescaled_energy, result.result[1]

#         return result

#     def set_backend(self, backend: QuantumBackend | None) -> None:
#         """Attach a QuantumBackend instance for execution."""
#         if backend is None:
#             backend = QiskitSimulatorBackend()
#         self.backend = backend
#         return

"""
Controlled time-evolved QSCI with spectral (tight) rescaling.

Summary of the convention
-------------------------
The importance filter applied by one-qubit phase estimation weights each
Hamiltonian eigenstate by

    w(lambda) = (1 + cos(phi(lambda))) / 2 ,

so the useful (monotonically decreasing) window is phi in [0, pi].  We
therefore map the physical spectrum into that window with an affine map
parameterised by a spectral *centre* ``c``, a spectral *half-width* ``S``
and a window fraction ``alpha``:

    phi(lambda) = (alpha * pi / (2 S)) * (lambda - c + S)                (*)

Writing ``t_eff = alpha * pi / (2 S)`` for the effective physical evolution
time, Equation (*) is realised by the controlled unitary

    exp(-i t_eff (H_nonid + g))  ,      g = c_0 - c + S ,

where ``H_nonid`` collects the non-identity Pauli terms of ``H`` and ``c_0``
is the coefficient of the identity string (0 if absent).  The constant ``g``
is *not* a global phase here: the operator is applied conditionally on the
ancilla, so ``g`` shifts the filter window and must be implemented.  It is
applied exactly, as a single phase gate ``p(-g * t_eff)`` on the ancilla.

Choosing (c, S)
---------------
Any (c, S) gives a valid filter.  Two properties depend on the choice:

* **Guarantee.**  If the spectrum is enclosed, ``spectrum`` subset of
  ``[c - S, c + S]``, then phi lies in ``[0, alpha*pi]`` and, for
  ``alpha <= 1/2``, every weight is non-negative, giving the unconditional
  floor ``P_surv >= 1/2``.  Enclosure is rigorous for the coefficient
  l1-norm choices ``(0, Lambda)`` and ``(c_0, Lambda')``; it is only as good
  as the estimate for spectral choices.
* **Strength.**  The filter resolves energies on the scale of ``S``.  With
  ``S = Lambda`` the real spectrum occupies only a fraction ``W / Lambda``
  of the window (typically 18-30% for Jordan-Wigner chemistry Hamiltonians),
  and both the filtering and the effective evolution time are suppressed by
  that factor.  Taking ``c`` and ``S`` from the actual spectral edges
  recovers the full window.

Independently of enclosure, the quadratic bound

    P_surv >= 1 - (alpha^2 pi^2 / (16 S^2)) [Var_psi(H) + (<H>_psi - c + S)^2]

holds for *every* (c, S), every alpha and every reference state, and is
numerically tight (gap < 1e-3) in the tight-normalisation regime.  It is
exposed as :meth:`ControlledTimeEvolvedQSCI.survival_bound`.

Changes relative to the previous implementation
-----------------------------------------------
1. ``duration`` is gone.  The window is set by ``alpha`` alone; the physical
   time follows as ``t_eff = alpha*pi/(2S)``.  Previously ``duration`` and
   the rescaling both carried a factor of the norm, which made the realised
   window ``alpha_eff = Lambda * t / pi`` rather than the intended one.
2. ``(c, S)`` are configurable, with spectral estimation available.
3. The constant offset is applied as an ancilla phase gate.  Previously it
   was pushed into the Hamiltonian dictionary as an all-identity Pauli
   string, which ``exp_pauli_string_to_circ`` silently maps to an *empty*
   circuit -- so the offset was never applied, and (under qDRIFT) roughly
   half of all sampled terms were no-ops.  Without the offset the filter
   weight is symmetric about ``c_0`` and suppresses large ``|lambda|``
   rather than large ``lambda``.
4. The subspace diagonalisation uses the physical Hamiltonian, so energies
   come back in Hartree with no affine inversion.
"""

from __future__ import annotations

import copy
import warnings
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit

from ...mps import MatrixProductState
from ...quantum_algorithms.hamiltonian_simulation.qdrift import (
    QDriftSimulation,
)
from ...quantum_algorithms.hamiltonian_simulation.trotterisation import (
    TrotterSimulation,
)
from ...tn_methods.mps_to_circuit import MPSAnalyticDecomposition
from ..backend.base import QuantumBackend
from ..backend.qiskit_simulator import QiskitSimulatorBackend
from ..base import QuantumAlgorithm
from ..result import Result
from ..utils import add_controls
from .qsci import QSCI

#: Default qubit ceiling for exact (Lanczos) spectral-edge estimation.
LANCZOS_MAX_QUBITS = 18

#: Alpha above which the unconditional survival floor no longer applies.
ALPHA_GUARANTEE = 0.5


class ControlledTimeEvolvedQSCI(QuantumAlgorithm):
    """QSCI with an ancilla-controlled time evolution acting as an importance
    filter on the sampled configurations.

    Args:
        hamiltonian: Qubit Hamiltonian as ``{pauli_string: coefficient}``.
        reference_state: Reference state as an MPS.
        alpha: Fraction of the usable phase window ``[0, pi]`` that the
            spectrum is mapped onto.  ``alpha <= 1/2`` carries the
            unconditional survival floor of one half (given enclosure);
            larger ``alpha`` filters more aggressively and relies on the
            quadratic bound instead.  Values above
            ``2S / (lambda_max - c + S)`` invert the filter and are refused.
        num_circuits: Number of circuits in the phase ladder.  Circuit ``k``
            (1-indexed) uses ``alpha_k = k * alpha / num_circuits``, so the
            deepest circuit realises the requested ``alpha`` and the
            guarantee holds for every member of the ladder.
        spectral_range: Explicit ``(lambda_min, lambda_max)`` enclosure.  If
            given, ``spectral_estimator`` is ignored.
        spectral_estimator: How to obtain the spectral edges.

            * ``"lanczos"`` -- matrix-free Lanczos on both edges.  Exact, and
              used only up to ``lanczos_max_qubits`` qubits, beyond which it
              falls back to ``"l1"`` with a warning.
            * ``"dmrg"`` -- DMRG on ``H`` and ``-H``.  Scalable but
              *variational*, so the estimated width is a lower bound on the
              true width; enclosure is then not rigorous and the floor
              becomes conditional.
            * ``"l1"`` -- rigorous and estimation-free: ``c = c_0``,
              ``S = Lambda'`` (the identity term is removed from the norm,
              which is free and slightly tighter than ``S = Lambda``).
            * ``"l1_total"`` -- the original convention, ``c = 0``,
              ``S = Lambda``.  Retained for reproducing earlier results.
        spectral_padding: Fractional padding applied to an *estimated*
            half-width, ``S = (1 + padding) * W``.  Ignored for the ``l1``
            estimators, which are already rigorous.
        lanczos_max_qubits: Qubit ceiling for the Lanczos estimator.
        diagonalise_rescaled: If ``True``, project the rescaled operator
            rather than the physical Hamiltonian and invert the affine map
            afterwards.  Only for backwards comparison; the default
            ``False`` is numerically preferable.
    """

    def __init__(
        self,
        hamiltonian: dict,
        reference_state: MatrixProductState,
        *,
        alpha: float = 0.5,
        num_circuits: int = 5,
        spectral_range: tuple[float, float] | None = None,
        spectral_estimator: str = "lanczos",
        spectral_padding: float = 0.1,
        lanczos_max_qubits: int = LANCZOS_MAX_QUBITS,
        backend: QuantumBackend | None = None,
        qdrift: bool = True,
        num_qdrift_circuits: int | None = 10,
        qdrift_error: float | None = None,
        max_qdrift_terms: int | None = 800,
        num_electrons: int | None = None,
        known_important_configurations: list[str] = [],
        known_unimportant_configurations: list[str] = [],
        postprocessing_function: list[Callable] | None = None,
        postprocessing_args: list[dict] | None = None,
        diagonalise_rescaled: bool = False,
        num_alpha: int | None = None,
        num_beta: int | None = None,
        spin_orbital_order: str | None = None,
        verify_spin: bool = True,
    ) -> ControlledTimeEvolvedQSCI:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if num_circuits < 1:
            raise ValueError("num_circuits must be at least 1.")
        if spectral_padding < 0.0:
            raise ValueError("spectral_padding must be non-negative.")

        # ---- physical Hamiltonian and its norms -------------------------
        self._pauli_table_cache = None
        self.hamiltonian_physical = self.sanitize_dict(hamiltonian)
        self.num_qubits = len(next(iter(self.hamiltonian_physical)))
        identity = "I" * self.num_qubits

        self.identity_coeff = float(self.hamiltonian_physical.get(identity, 0.0))
        self.hamiltonian_nonidentity = {
            k: v
            for k, v in self.hamiltonian_physical.items()
            if k != identity and abs(v) > 1e-14
        }
        if not self.hamiltonian_nonidentity:
            raise ValueError(
                "Hamiltonian has no non-identity terms; there is nothing to "
                "evolve under."
            )

        self.l1_norm = float(sum(abs(v) for v in self.hamiltonian_physical.values()))
        self.l1_norm_nonidentity = float(
            sum(abs(v) for v in self.hamiltonian_nonidentity.values())
        )
        #: Retained for backwards compatibility with the old attribute name.
        self.norm = self.l1_norm

        # ---- choose the affine map --------------------------------------
        self.spectral_estimator = spectral_estimator
        self.spectral_padding = float(spectral_padding)
        self.lanczos_max_qubits = int(lanczos_max_qubits)
        (
            self.centre,
            self.half_width,
            self.spectral_edges,
            self.enclosure_is_rigorous,
            self.rescaling_scheme,
        ) = self._choose_affine_map(spectral_range)

        # ---- alpha ceiling: above it the filter inverts ------------------
        self.alpha_ceiling = self._alpha_ceiling()
        if self.alpha_ceiling is not None and alpha > self.alpha_ceiling + 1e-12:
            raise ValueError(
                f"alpha = {alpha:.4f} exceeds the ceiling "
                f"{self.alpha_ceiling:.4f} = 2S / (lambda_max - c + S) for this "
                "Hamiltonian and rescaling: the top of the spectrum would be "
                "pushed past phase pi, where the filter weight increases again "
                "with energy and the importance ordering is inverted."
            )
        self.alpha = float(alpha)

        if self.alpha > ALPHA_GUARANTEE + 1e-12:
            warnings.warn(
                f"alpha = {self.alpha:.3f} > 1/2: the unconditional survival "
                "floor of 1/2 no longer applies. Use survival_bound() for the "
                "state-dependent guarantee, which remains valid.",
                stacklevel=2,
            )
        if not self.enclosure_is_rigorous:
            warnings.warn(
                f"The '{self.rescaling_scheme}' rescaling uses estimated "
                "spectral edges, so spectrum-in-window is not guaranteed and "
                "the 1/2 floor is conditional on the estimate. The quadratic "
                "bound from survival_bound() holds regardless.",
                stacklevel=2,
            )

        # ---- derived evolution parameters --------------------------------
        #: Effective physical evolution time of the deepest ladder circuit.
        self.t_eff = self.alpha * np.pi / (2.0 * self.half_width)
        #: Coefficient of the identity in the shifted operator.
        self.offset_coeff = self.identity_coeff - self.centre + self.half_width

        self.num_circuits = int(num_circuits)
        #: Physical evolution times of the ladder, shallowest first.
        self.evolution_times = [
            (k + 1) * self.t_eff / self.num_circuits for k in range(self.num_circuits)
        ]
        #: Window fraction realised by each ladder circuit.
        self.alpha_ladder = [
            (k + 1) * self.alpha / self.num_circuits for k in range(self.num_circuits)
        ]

        # ---- the operator handed to the simulation routines --------------
        # Only the non-identity terms; the offset is a gate on the ancilla.
        self.hamiltonian_evolution = dict(self.hamiltonian_nonidentity)
        #: Rescaled operator with the offset folded back in. Not used for
        #: circuit construction; kept so that ``rescale_energy`` has a
        #: well-defined inverse and for diagnostic comparison.
        self.hamiltonian = self._rescaled_hamiltonian()

        # ---- the rest of the pipeline ------------------------------------
        self.circuits: list[QuantumCircuit] = []
        self.reference_state = reference_state
        self.reference_state_qc = self.create_reference_circuit()
        self.backend = self.set_backend(backend=backend)

        self.qdrift = qdrift
        self.num_qdrift_circuits = num_qdrift_circuits
        self.qdrift_error = qdrift_error
        if qdrift and max_qdrift_terms is None:
            raise ValueError(
                "max_qdrift_terms must be set when qdrift=True "
                "(QDriftSimulation takes min(num_terms, max_num_terms))."
            )
        self.max_qdrift_terms = max_qdrift_terms

        self.num_electrons = num_electrons
        self.num_alpha = num_alpha
        self.num_beta = num_beta
        self.spin_orbital_order = spin_orbital_order
        self.verify_spin = verify_spin
        self.important_configurations = known_important_configurations
        self.unimportant_configurations = known_unimportant_configurations
        self.postprocessing_function = postprocessing_function
        self.postprocessing_args = postprocessing_args
        self.diagonalise_rescaled = bool(diagonalise_rescaled)

        self.circuit_depths: list[int] = []
        self.energy = None
        #: Survival statistics of the most recent ``run``.
        self.survival_rate = None

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def circuit(self) -> QuantumCircuit:
        """Get the circuit from the algorithm"""
        return self.circuits

    @property
    def max_evolution_time(self) -> float:
        """Physical time of the deepest circuit; a proxy for circuit depth."""
        return self.evolution_times[-1]

    @property
    def total_evolution_time(self) -> float:
        """Summed physical time over the ladder; the matched-comparison
        quantity against TE-QSCI, which builds the same style of ladder."""
        return float(np.sum(self.evolution_times))

    @property
    def window_fraction(self) -> float:
        """Fraction of the phase window ``[0, alpha*pi]`` that the true
        spectrum occupies.  One for an exactly tight rescaling."""
        if self.spectral_edges is None:
            return float("nan")
        lo, hi = self.spectral_edges
        return float((hi - lo) / (2.0 * self.half_width))

    @property
    def discrimination(self) -> float:
        """Ratio of the filter weights at the bottom and top spectral edges.

        For an estimated rescaling with padding ``p`` and window fraction
        ``alpha`` this is a function of ``alpha`` and ``p`` alone, which makes
        it the natural design dial:
        ``[1 + cos(alpha*pi*p/(2(1+p)))] / [1 + cos(alpha*pi*(2+p)/(2(1+p)))]``.
        """
        if self.spectral_edges is None:
            return float("nan")
        lo, hi = self.spectral_edges
        w_lo = self.filter_weight(lo)
        w_hi = self.filter_weight(hi)
        return float("inf") if w_hi <= 1e-14 else float(w_lo / w_hi)

    # ------------------------------------------------------------------ #
    # Rescaling                                                           #
    # ------------------------------------------------------------------ #

    def sanitize_dict(self, d: dict[str, complex | float]) -> dict[str, float]:
        return {
            k: float(v.real) if isinstance(v, complex) else float(v)
            for k, v in d.items()
        }

    def _choose_affine_map(
        self, spectral_range: tuple[float, float] | None
    ) -> tuple[float, float, tuple[float, float] | None, bool, str]:
        """Return ``(centre, half_width, edges, rigorous, scheme_name)``."""
        if spectral_range is not None:
            lo, hi = float(spectral_range[0]), float(spectral_range[1])
            if hi <= lo:
                raise ValueError("spectral_range must satisfy hi > lo.")
            return (*self._pad(lo, hi), (lo, hi), True, "given")

        estimator = self.spectral_estimator
        if estimator == "l1_total":
            return 0.0, self.l1_norm, None, True, "l1_total"
        if estimator == "l1":
            return self.identity_coeff, self.l1_norm_nonidentity, None, True, "l1"

        if estimator == "lanczos":
            if self.num_qubits > self.lanczos_max_qubits:
                warnings.warn(
                    f"{self.num_qubits} qubits exceeds lanczos_max_qubits="
                    f"{self.lanczos_max_qubits}; falling back to the rigorous "
                    "'l1' rescaling. Pass spectral_range=... or use "
                    "spectral_estimator='dmrg' for a tighter window.",
                    stacklevel=3,
                )
                return (
                    self.identity_coeff,
                    self.l1_norm_nonidentity,
                    None,
                    True,
                    "l1",
                )
            lo, hi = self._lanczos_edges()
            return (*self._pad(lo, hi), (lo, hi), True, "lanczos")

        if estimator == "dmrg":
            lo, hi = self._dmrg_edges()
            return (*self._pad(lo, hi), (lo, hi), False, "dmrg")

        raise ValueError(
            f"Unknown spectral_estimator '{estimator}'. Choose from "
            "'lanczos', 'dmrg', 'l1', 'l1_total'."
        )

    def _pad(self, lo: float, hi: float) -> tuple[float, float]:
        """Centre and padded half-width for a pair of spectral edges.

        The padded half-width is clipped at the rigorous ``l1`` value so a
        large padding can never make the window looser than the
        estimation-free choice.
        """
        centre = 0.5 * (hi + lo)
        half_width = 0.5 * (hi - lo) * (1.0 + self.spectral_padding)
        rigorous_half_width = self.l1_norm_nonidentity + abs(
            centre - self.identity_coeff
        )
        return centre, float(min(half_width, rigorous_half_width))

    def _rescaled_hamiltonian(self) -> dict[str, float]:
        """The shifted operator ``H - c + S``, whose evolution for time
        ``t_eff`` realises Equation (*).

        Only the identity coefficient differs from the physical Hamiltonian;
        the shift is entirely a constant.  Kept so ``rescale_energy`` has a
        well-defined inverse and for diagnostics.
        """
        out = dict(self.hamiltonian_nonidentity)
        out["I" * self.num_qubits] = self.offset_coeff
        return out

    def _alpha_ceiling(self) -> float | None:
        """Largest ``alpha`` keeping the whole spectrum below phase ``pi``.

        Beyond it the filter weight starts increasing with energy again, so
        the top of the spectrum is *preferred*.  Only computable when the
        spectral edges are known; the ``l1`` schemes are safe for all
        ``alpha <= 1`` by construction, so ``None`` is returned.
        """
        if self.spectral_edges is None:
            return None
        hi = self.spectral_edges[1]
        span = hi - self.centre + self.half_width
        return float(2.0 * self.half_width / span) if span > 0 else None

    def rescale_energy(self, energy_prime: float) -> float:
        """Invert the affine map, ``lambda = energy_prime + c - S``.

        Only needed when ``diagonalise_rescaled=True``.
        """
        return float(energy_prime + self.centre - self.half_width)

    # ------------------------------------------------------------------ #
    # Spectral edge estimators                                            #
    # ------------------------------------------------------------------ #

    def _pauli_tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Index-arithmetic description of the Hamiltonian's Pauli strings.

        A Pauli string ``P`` acts on a computational basis state as

            ``P|j> = i^{n_Y} (-1)^{popcount(j & z_mask)} |j XOR flip_mask>``

        with ``flip_mask`` set by the X and Y sites and ``z_mask`` by the Y and
        Z sites.  Site ``i`` of the string acts on qubit ``i``, matching the
        bitstring convention used throughout the class (site 0 first).

        Returns:
            ``(flip_masks, z_masks, y_phases, coeffs)``.
        """
        if getattr(self, "_pauli_table_cache", None) is not None:
            return self._pauli_table_cache

        n = self.num_qubits
        flips, zs, phases, cs = [], [], [], []
        for pauli, coeff in self.hamiltonian_physical.items():
            if coeff == 0.0:
                continue
            flip = 0
            z_mask = 0
            for i, p in enumerate(pauli):
                bit = 1 << (n - 1 - i)
                if p in ("X", "Y"):
                    flip |= bit
                if p in ("Y", "Z"):
                    z_mask |= bit
            flips.append(flip)
            zs.append(z_mask)
            phases.append((1j) ** pauli.count("Y"))
            cs.append(coeff)

        self._pauli_table_cache = (
            np.asarray(flips, dtype=np.int64),
            np.asarray(zs, dtype=np.int64),
            np.asarray(phases, dtype=np.complex128),
            np.asarray(cs, dtype=np.float64),
        )
        return self._pauli_table_cache

    @staticmethod
    def _popcount(x: np.ndarray) -> np.ndarray:
        """Bit population count for a non-negative integer array."""
        return (
            np.bitwise_count(x)
            if hasattr(np, "bitwise_count")
            else np.vectorize(lambda v: bin(int(v)).count("1"))(x)
        )

    def _lanczos_edges(self) -> tuple[float, float]:
        """Exact extreme eigenvalues via matrix-free Lanczos."""
        from scipy.sparse.linalg import LinearOperator, eigsh

        dim = 2**self.num_qubits
        flips, zs, phases, cs = self._pauli_tables()
        idx = np.arange(dim, dtype=np.int64)

        def matvec(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=np.complex128).reshape(-1)
            out = np.zeros(dim, dtype=np.complex128)
            for flip, z_mask, phase, coeff in zip(flips, zs, phases, cs):
                src = idx ^ flip
                amp = coeff * phase * v[src]
                if z_mask:
                    # Phase depends on the *source* index, which differs from
                    # the target index at every flipped site.
                    par = self._popcount(src & z_mask) & 1
                    amp = np.where(par == 1, -amp, amp)
                out += amp
            return out

        op = LinearOperator((dim, dim), matvec=matvec, dtype=np.complex128)
        kwargs = dict(k=1, return_eigenvectors=False, tol=1e-8, maxiter=10000)
        lo = float(np.real(eigsh(op, which="SA", **kwargs)[0]))
        hi = float(np.real(eigsh(op, which="LA", **kwargs)[0]))
        return lo, hi

    def _dmrg_edges(self, max_bond: int = 32, nsweeps: int = 12) -> tuple[float, float]:
        """Variational edge estimates from DMRG on ``H`` and ``-H``.

        Both are variational, so the returned interval is *contained* in the
        true spectral interval.  ``spectral_padding`` compensates; the
        survival floor is nonetheless only conditional.
        """
        from ...dmrg import DMRG

        lo, _ = DMRG(self.hamiltonian_physical, max_mps_bond=max_bond).run(nsweeps)
        neg = {k: -v for k, v in self.hamiltonian_physical.items()}
        hi_neg, _ = DMRG(neg, max_mps_bond=max_bond).run(nsweeps)
        return float(lo), float(-hi_neg)

    # ------------------------------------------------------------------ #
    # Survival diagnostics                                                #
    # ------------------------------------------------------------------ #

    def filter_weight(self, energy: float, alpha: float | None = None) -> float:
        """Post-selection weight ``(1 + cos(phi)) / 2`` of an eigenstate."""
        a = self.alpha if alpha is None else alpha
        phi = (a * np.pi / (2.0 * self.half_width)) * (
            energy - self.centre + self.half_width
        )
        return float(0.5 * (1.0 + np.cos(phi)))

    def survival_floor(self, alpha: float | None = None) -> float | None:
        """Unconditional floor on ``P_surv``, or ``None`` where none holds.

        Returns ``0.5`` when ``alpha <= 1/2`` *and* the window provably
        encloses the spectrum.  Returns ``None`` otherwise, so a caller
        cannot silently plot a floor that does not exist.
        """
        a = self.alpha if alpha is None else alpha
        if a <= ALPHA_GUARANTEE + 1e-12 and self.enclosure_is_rigorous:
            return 0.5
        return None

    def hamiltonian_moments(self, bitstring: str) -> tuple[float, float]:
        """``(<H>, Var(H))`` for a computational basis state, in ``O(L)``.

        ``H|j>`` is a sum over Pauli strings of amplitudes landing on
        ``|j XOR flip_mask>``.  Several strings can share a flip mask -- the
        ``XX`` and ``YY`` partners of a real Hamiltonian always do -- and
        their amplitudes interfere, so the contributions must be *accumulated
        per output configuration* before squaring.  Summing squared
        coefficients term by term overestimates the variance.
        """
        if len(bitstring) != self.num_qubits:
            raise ValueError(
                f"bitstring has length {len(bitstring)}, expected "
                f"{self.num_qubits}."
            )
        flips, zs, phases, cs = self._pauli_tables()
        j = int(bitstring, 2)

        parity = self._popcount(np.int64(j) & zs) & 1
        amps = cs * phases * np.where(parity == 1, -1.0 + 0j, 1.0 + 0j)
        outputs = np.int64(j) ^ flips

        unique_out, inverse = np.unique(outputs, return_inverse=True)
        acc = np.zeros(unique_out.size, dtype=np.complex128)
        np.add.at(acc, inverse, amps)

        diag = unique_out == j
        mean = float(np.real(acc[diag].sum())) if diag.any() else 0.0
        var = float(np.sum(np.abs(acc[~diag]) ** 2))
        return mean, var

    def survival_bound(
        self,
        reference: str | tuple[float, float],
        alpha: float | None = None,
    ) -> float:
        """State-dependent lower bound on ``P_surv``.

        ``P_surv >= 1 - (alpha^2 pi^2 / 16 S^2) [Var(H) + (<H> - c + S)^2]``

        Follows from ``cos(x) >= 1 - x^2/2`` applied to the phase operator, so
        it needs no enclosure assumption and holds for any ``(c, S)`` and any
        ``alpha``.

        Args:
            reference: Either a computational basis bitstring, or an explicit
                ``(mean, variance)`` pair for a general reference state.
            alpha: Window fraction; defaults to the instance value.
        """
        a = self.alpha if alpha is None else alpha
        if isinstance(reference, str):
            mean, var = self.hamiltonian_moments(reference)
        else:
            mean, var = float(reference[0]), float(reference[1])
        shifted = mean - self.centre + self.half_width
        return float(
            1.0 - (a**2 * np.pi**2 / (16.0 * self.half_width**2)) * (var + shifted**2)
        )

    def summary(self) -> dict:
        """Everything a caller needs to label a plot or audit a run."""
        return {
            "scheme": self.rescaling_scheme,
            "alpha": self.alpha,
            "centre": self.centre,
            "half_width": self.half_width,
            "spectral_edges": self.spectral_edges,
            "enclosure_is_rigorous": self.enclosure_is_rigorous,
            "alpha_ceiling": self.alpha_ceiling,
            "t_eff": self.t_eff,
            "offset_coeff": self.offset_coeff,
            "offset_share_of_l1": abs(self.offset_coeff)
            / (self.l1_norm_nonidentity + abs(self.offset_coeff)),
            "l1_norm": self.l1_norm,
            "l1_norm_nonidentity": self.l1_norm_nonidentity,
            "window_fraction": self.window_fraction,
            "discrimination": self.discrimination,
            "num_circuits": self.num_circuits,
            "evolution_times": list(self.evolution_times),
            "max_evolution_time": self.max_evolution_time,
            "total_evolution_time": self.total_evolution_time,
            "survival_floor": self.survival_floor(),
        }

    # ------------------------------------------------------------------ #
    # Circuit construction                                                #
    # ------------------------------------------------------------------ #

    def create_reference_circuit(self) -> QuantumCircuit:
        """Create a circuit to prepare the reference state"""
        if self.reference_state.bond_dimension == 1:
            mpstocirc = MPSAnalyticDecomposition(self.reference_state, 1, 1.0)
            qc = mpstocirc.bond_dim_2_to_qc_exact(self.reference_state)
        else:
            self.reference_state.compress(2)
            mpstocirc = MPSAnalyticDecomposition(self.reference_state, 1, 1.0)
            qc = mpstocirc.mps_to_qc_via_ttn(self.reference_state, 2)
        return qc

    def _assemble(self, sim_circ: QuantumCircuit, duration: float) -> QuantumCircuit:
        """Wrap a bare evolution circuit into the interferometer.

        ``H_a`` -> controlled evolution -> offset phase on the ancilla ->
        ``H_a``.  The offset is the identity part of the shifted operator; the
        controlled version of ``exp(-i g t I)`` is exactly ``p(-g t)`` on the
        ancilla, so no multi-qubit control is needed and the offset is applied
        without Trotter or qDRIFT error.
        """
        controlled = QuantumCircuit(sim_circ.num_qubits + 1)
        controlled.compose(
            sim_circ, qubits=range(1, sim_circ.num_qubits + 1), inplace=True
        )
        controlled = add_controls(controlled, [0])

        ref_copy = copy.deepcopy(self.reference_state_qc)
        qc = QuantumCircuit(ref_copy.num_qubits + 1)
        qc.h(0)
        qc.compose(ref_copy, qubits=range(1, ref_copy.num_qubits + 1), inplace=True)
        qc.compose(controlled, inplace=True)
        qc.p(-self.offset_coeff * duration, 0)
        qc.h(0)
        return qc

    def perform_controlled_time_evolution(self, duration: float) -> QuantumCircuit:
        """Controlled Trotterised evolution for a given physical time."""
        if duration == 0.0:
            return copy.deepcopy(self.reference_state_qc)
        sim = TrotterSimulation(self.hamiltonian_evolution, duration=duration)
        return self._assemble(sim.circuit, duration)

    def perform_controlled_time_evolution_qdrift(
        self, duration: float, error: float | None = None
    ) -> QuantumCircuit:
        """Controlled qDRIFT evolution for a given physical time."""
        if duration == 0.0:
            return copy.deepcopy(self.reference_state_qc)
        sim = QDriftSimulation(
            self.hamiltonian_evolution,
            duration=duration,
            error=error,
            max_num_terms=self.max_qdrift_terms,
        )
        return self._assemble(sim.circuit, duration)

    def build_circuits_trotter(self) -> list[QuantumCircuit]:
        """Trotterised phase ladder."""
        return [self.perform_controlled_time_evolution(t) for t in self.evolution_times]

    def build_circuits_qdrift(self) -> list[QuantumCircuit]:
        """qDRIFT phase ladder, with several stochastic realisations per rung."""
        circuits = []
        for t in self.evolution_times:
            for _ in range(self.num_qdrift_circuits):
                circuits.append(
                    self.perform_controlled_time_evolution_qdrift(
                        t, error=self.qdrift_error
                    )
                )
        return circuits

    def build_circuits(self) -> list[QuantumCircuit]:
        self.circuits = (
            self.build_circuits_qdrift()
            if self.qdrift
            else self.build_circuits_trotter()
        )
        self.circuit_depths = [qc.depth() for qc in self.circuits]
        return self.circuits

    # ------------------------------------------------------------------ #
    # Post-selection and execution                                        #
    # ------------------------------------------------------------------ #

    def post_selection(self, counts: dict[str, int]) -> dict[str, int]:
        """Keep only shots with the ancilla in ``|0>`` and record the rate.

        The backend returns bitstrings with site 0 first, so the ancilla is
        the leading character.
        """
        new_counts: dict[str, int] = {}
        kept = 0
        total = 0
        for b, count in counts.items():
            if len(b) == self.reference_state.num_sites:
                # No ancilla present (duration 0 circuits).
                new_counts[b] = new_counts.get(b, 0) + count
                kept += count
                total += count
                continue
            total += count
            if b[0] == "0":
                new_counts[b[1:]] = new_counts.get(b[1:], 0) + count
                kept += count
        self.survival_rate = (kept / total) if total else None
        return new_counts

    def run(self, num_shots: int, subspace_size: int) -> Result:
        """Run the full algorithm pipeline. Returns a Result object."""
        circuits = self.build_circuits()
        if self.postprocessing_function is not None:
            postprocessing_funcs = [self.post_selection] + list(
                self.postprocessing_function
            )
            func_args = [None] + list(self.postprocessing_args)
        else:
            postprocessing_funcs = [self.post_selection]
            func_args = [None]

        diag_hamiltonian = (
            self.hamiltonian if self.diagonalise_rescaled else self.hamiltonian_physical
        )
        qsci = QSCI(
            circuits,
            self.backend,
            diag_hamiltonian,
            self.num_electrons,
            self.important_configurations,
            self.unimportant_configurations,
            postprocessing_funcs,
            func_args,
            self.num_alpha,
            self.num_beta,
            self.spin_orbital_order,
            self.verify_spin,
        )
        result = qsci.run(num_shots, subspace_size)

        if self.diagonalise_rescaled:
            result.result = self.rescale_energy(result.result[0]), result.result[1]
        self.energy = result.result[0]

        result.metadata["survival_rate"] = self.survival_rate
        result.metadata["rescaling"] = self.summary()
        return result

    def set_backend(self, backend: QuantumBackend | None) -> None:
        """Attach a QuantumBackend instance for execution."""
        if backend is None:
            backend = QiskitSimulatorBackend()
        self.backend = backend
        return
