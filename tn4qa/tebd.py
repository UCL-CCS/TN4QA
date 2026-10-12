from __future__ import annotations

import copy

import numpy as np

from .mpo import MatrixProductOperator
from .mps import MatrixProductState


class TEBD:
    """
    Time-Evolving Block Decimation for a Hamiltonian given as a weighted sum
    of Pauli strings, H = sum_k w_k P_k.

    Each Trotter factor exp(-i * w_k * P_k * dt) is applied to the MPS as an
    exact bond-dimension-2 MPO (see pauli_exponential_mpo), followed by
    compression back to the maximum bond dimension. This handles arbitrary
    (long-range / all-to-all) Pauli strings directly, with no requirement of
    nearest-neighbour structure -- important for chemistry Hamiltonians,
    whose Jordan-Wigner-mapped terms carry long Z-strings.

    Interface mirrors the DMRG class: a Pauli-dict Hamiltonian, a maximum
    MPS bond dimension, and an optional initial MPS.
    """

    def __init__(
        self,
        hamiltonian: dict[str, complex],
        max_mps_bond: int = 16,
        initial_mps: MatrixProductState | None = None,
    ) -> TEBD:
        self.hamiltonian = hamiltonian
        pauli = {
            key: value for key, value in hamiltonian.items() if np.abs(value) > 1e-12
        }
        self.pauli_terms = list(pauli.items())
        self.nqubits = len(self.pauli_terms[0][0])
        self.max_bond = max_mps_bond

        for p_string, weight in self.pauli_terms:
            if np.abs(np.imag(weight)) > 1e-10:
                raise ValueError(
                    f"Hamiltonian coefficient for {p_string} is not real: {weight}"
                )

        if initial_mps is None:
            initial_mps = MatrixProductState.all_zero_mps(self.nqubits)
        self.initial_state = initial_mps
        self.mps = copy.deepcopy(initial_mps)
        self.times = [0.0]
        return

    def _term_mpos(self, dt: float) -> list[MatrixProductOperator]:
        """Precompute the per-term Trotter-factor MPOs for a step of size dt.

        exp(-i * w * P * dt) = exp(i * x * P) with x = -w * dt.
        """
        return [
            MatrixProductOperator.from_pauli_exponential(
                p_string, -np.real(weight) * dt
            )
            for p_string, weight in self.pauli_terms
        ]

    def _apply_term_sequence(
        self, mpos: list[MatrixProductOperator], reverse: bool = False
    ) -> None:
        seq = mpos[::-1] if reverse else mpos
        for term_mpo in seq:
            self.mps = self.mps.apply_mpo(term_mpo)
            if self.mps.bond_dimension > self.max_bond:
                self.mps.compress(self.max_bond)
            self.mps.normalise()

    def run(
        self,
        time: float,
        trotter_steps: int,
        order: int = 2,
        callback=None,
    ) -> MatrixProductState:
        """
        Evolve the MPS to exp(-i * H * time) |initial> via Trotterisation.

        Args:
            time: Total evolution time t.
            trotter_steps: Number of Trotter steps N_T (step size dt = t / N_T).
            order: Trotter order, 1 (product in forward order) or 2
                (symmetric: half-step forward order, half-step reverse order).
            callback: Optional callable f(step_index, current_time, mps),
                invoked after every completed Trotter step.

        Returns:
            The evolved MPS (also stored on self.mps).
        """
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")

        dt = time / trotter_steps
        if order == 1:
            step_mpos = self._term_mpos(dt)
        else:
            half_step_mpos = self._term_mpos(dt / 2)

        for step in range(trotter_steps):
            if order == 1:
                self._apply_term_sequence(step_mpos)
            else:
                self._apply_term_sequence(half_step_mpos)
                self._apply_term_sequence(half_step_mpos, reverse=True)
            self.times.append(self.times[-1] + dt)
            if callback is not None:
                callback(step, self.times[-1], self.mps)

        return self.mps
