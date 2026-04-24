from __future__ import annotations

import copy

import numpy as np
import scipy.linalg
import sparse
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.quantum_info import Operator

from .mpo import MatrixProductOperator
from .mps import MatrixProductState
from .tensor import StorageHint, _as_dense


def _gate_key(inst: CircuitInstruction) -> str:  # type: ignore
    params = tuple(float(p) for p in inst.operation.params)
    return f"{inst.operation.name}_{params}"


_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)


def _apply_swap_dense(mps: MatrixProductState, site: int, max_bond: int | None) -> None:
    """Apply SWAP between sites site and site+1 in-place (dense SVD)."""
    _apply_local_gate_dense(mps, _SWAP, site, site + 1, max_bond)


def _apply_local_gate_dense(
    mps: MatrixProductState,
    gate: ndarray,  # (4, 4)
    site0: int,
    site1: int,
    max_bond: int | None,
    tol: float = 1e-12,
) -> None:
    """
    Apply a local 2-qubit gate to neighbouring sites (site0, site1 = site0+1).
    Dense throughout.  Updates MPS in-place.

    MPS index convention
    --------------------
    Site 1   (left boundary) : (down, phys)          axes (χ_d, d)
    Site N   (right boundary): (up,   phys)           axes (χ_u, d)
    Interior                 : (up,   down, phys)     axes (χ_u, χ_d, d)

    Gate G has shape (2, 2, 2, 2) = (out0, out1, in0, in1).
    out0/in0 act on site0, out1/in1 act on site1.
    The shared bond between t0 and t1 is t0's *down* index = t1's *up* index.

    SVD cut convention: U carries the t0 (left) tensor, Vh carries the t1 (right) tensor.
    The new virtual bond dimension is the singular-value index k.

    Post-SVD shapes
    ---------------
    site0==1, site1==N : t0 → (k, d) = (down, phys)
                         t1 → (k, d) = (up,   phys)
    site0==1           : t0 → (k, d)           = (down, phys)
                         t1 → (k, χ_d_t1, d)   = (up, down, phys)
    site1==N           : t0 → (χ_u_t0, k, d)   = (up, down, phys)
                         t1 → (k, d)            = (up, phys)
    interior           : t0 → (χ_u_t0, k, d)   = (up, down, phys)
                         t1 → (k, χ_d_t1, d)   = (up, down, phys)
    """
    assert site1 == site0 + 1
    n = mps.num_sites
    G = gate.reshape(2, 2, 2, 2)

    t0 = _as_dense(mps.tensors[site0 - 1].data)
    t1 = _as_dense(mps.tensors[site0].data)

    eps = 1e-14

    if site0 == 1 and site1 == n:
        merged = np.einsum("i,j,klij->kl", t0, t1, G)
        u, s, vh = scipy.linalg.svd(merged, full_matrices=False, check_finite=False)
        keep = _truncate_sv(
            s, min(2, 2) if max_bond is None else min(max_bond, 2, 2), tol
        )
        new_t0 = vh[:keep, :]
        new_t1 = np.moveaxis((u[:, :keep] * s[:keep]), 1, 0)

    elif site0 == 1:
        b = t1.shape[1]
        merged = np.einsum("ai,abj,klij->klb", t0, t1, G)
        mat = merged.transpose(0, 2, 1).reshape(2, b * 2)
        bond = min(max_bond, 2, b * 2) if max_bond else min(2, b * 2)
        u, s, vh = scipy.linalg.svd(mat, full_matrices=False, check_finite=False)
        keep = _truncate_sv(s, bond, tol)
        new_t0 = np.moveaxis((u[:, :keep] * s[:keep]), 1, 0)
        new_t1 = vh[:keep, :].reshape(keep, b, 2)

    elif site1 == n:
        a = t0.shape[0]
        merged = np.einsum("abi,bj,klij->akl", t0, t1, G)
        mat = merged.reshape(a * 2, 2)
        bond = min(max_bond, a * 2, 2) if max_bond else min(a * 2, 2)
        u, s, vh = scipy.linalg.svd(mat, full_matrices=False, check_finite=False)
        keep = _truncate_sv(s, bond, tol)
        new_t0 = (u[:, :keep] * s[:keep]).reshape(a, 2, keep)
        new_t0 = new_t0.transpose(0, 2, 1)
        new_t1 = vh[:keep, :]

    else:
        a, c = t0.shape[0], t1.shape[1]
        merged = np.einsum("abi,bcj,klij->aklc", t0, t1, G)
        mat = merged.reshape(a * 2, 2 * c)
        bond = min(max_bond, a * 2, 2 * c) if max_bond else min(a * 2, 2 * c)
        u, s, vh = scipy.linalg.svd(mat, full_matrices=False, check_finite=False)
        keep = _truncate_sv(s, bond, tol)
        new_t0 = (u[:, :keep] * s[:keep]).reshape(a, 2, keep)
        new_t0 = new_t0.transpose(0, 2, 1)
        new_t1 = vh[:keep, :].reshape(keep, 2, c)
        new_t1 = new_t1.transpose(0, 2, 1)

    new_t0[np.abs(new_t0) < eps] = 0.0
    new_t1[np.abs(new_t1) < eps] = 0.0

    mps.tensors[site0 - 1].data = new_t0
    mps.tensors[site0 - 1].dimensions = new_t0.shape
    mps.tensors[site0].data = new_t1
    mps.tensors[site0].dimensions = new_t1.shape

    mps.update_bond_information()


def _truncate_sv(s: ndarray, max_bond: int, tol: float) -> int:
    s_sq = s**2
    cum = np.cumsum(s_sq[::-1])[::-1]
    keep = len(s)
    for k in range(len(s)):
        if s[k] < 1e-14:
            keep = k
            break
        if cum[k] < tol**2:
            keep = k + 1
            break
    return max(1, min(keep, max_bond))


class CircuitSimulator:
    """
    Simulate a Qiskit QuantumCircuit using an MPS representation.
    All internal tensors are stored as dense ndarrays.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        input_state: MatrixProductState | None = None,
    ) -> None:
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self._set_input_state(input_state)
        self.current_state = copy.deepcopy(self.input_state)
        self.output_state = None
        self.mpo = MatrixProductOperator.identity_mpo(self.num_qubits)

        # Precompute gate matrices for the circuit (avoids repeated Operator calls)
        self._gate_matrices: dict[str, ndarray] = {}
        for inst in circuit.data:
            key = _gate_key(inst)
            if key not in self._gate_matrices:
                self._gate_matrices[key] = np.asarray(
                    Operator(inst.operation).reverse_qargs().data
                )

    def _set_input_state(self, inp: MatrixProductState | None) -> None:
        if inp is None:
            inp = MatrixProductState.all_zero_mps(self.num_qubits)
        # Ensure all tensors are dense
        for t in inp.tensors:
            if (
                t.is_sparse()
                if hasattr(t, "is_sparse")
                else isinstance(t.data, sparse.COO)
            ):
                t.data = _as_dense(t.data)
            t.storage_hint = StorageHint.DENSE
        self.input_state = inp
        self.input_state.set_default_indices()

    def apply_one_qubit_gate(self, mat: ndarray, site: int) -> None:
        t = self.current_state.tensors[site - 1]
        tdata = _as_dense(t.data)
        if site == 1 or site == self.num_qubits:
            result = np.einsum("ij,kj->ik", tdata, mat)
        else:
            result = np.einsum("ijk,lk->ijl", tdata, mat)
        t.data = result
        t.dimensions = result.shape

    def apply_local_two_qubit_gate(
        self,
        mat: ndarray,
        sites: list[int],
        max_bond: int | None = None,
        tol: float = 1e-12,
    ) -> None:
        site0, site1 = sites
        if mat.shape == (4, 4):
            gate = mat
        else:
            gate = mat.reshape(4, 4)
        if site1 < site0:
            # Flip qubit ordering
            gate = gate.reshape(2, 2, 2, 2)
            gate = np.moveaxis(gate, [0, 1, 2, 3], [1, 0, 3, 2]).reshape(4, 4)
            site0, site1 = site1, site0
        assert site1 == site0 + 1
        _apply_local_gate_dense(self.current_state, gate, site0, site1, max_bond, tol)

    def apply_nonlocal_two_qubit_gate(
        self,
        mat: ndarray,
        sites: list[int],
        max_bond: int | None = None,
    ) -> None:
        """
        Apply a gate between non-neighbouring qubits using a SWAP network.

        SWAP the target qubit adjacent to the control, apply the gate,
        then SWAP back.  Cost: O(|q1 - q0|) local two-qubit gates rather
        than a full-length MPO contraction.
        """
        site0, site1 = sorted(sites)
        if site1 == site0 + 1:
            self.apply_local_two_qubit_gate(mat, sites, max_bond)
            return

        # If the original ordering was (site1, site0), note that we need to
        # swap the qubit indices in the gate matrix after moving
        flipped = sites[0] > sites[1]

        # Move site1 qubit left to be adjacent to site0 via SWAPs
        for s in range(site1, site0 + 1, -1):
            _apply_swap_dense(self.current_state, s - 1, max_bond)

        # Apply gate at (site0, site0+1)
        gate = mat.reshape(4, 4)
        if flipped:
            gate = gate.reshape(2, 2, 2, 2)
            gate = np.moveaxis(gate, [0, 1, 2, 3], [1, 0, 3, 2]).reshape(4, 4)
        self.apply_local_two_qubit_gate(gate, [site0, site0 + 1], max_bond)

        # Swap back
        for s in range(site0 + 1, site1):
            _apply_swap_dense(self.current_state, s, max_bond)

    def run(
        self,
        max_bond_dimension: int | None = None,
        samples: int | None = None,
    ) -> MatrixProductState | dict:
        for inst in self.circuit.data:
            qidxs = [
                inst.qubits[i]._index + 1 for i in range(inst.operation.num_qubits)
            ]
            key = _gate_key(inst)
            mat = self._gate_matrices[key]

            if len(qidxs) == 1:
                self.apply_one_qubit_gate(mat, qidxs[0])
            elif len(qidxs) == 2:
                self.apply_nonlocal_two_qubit_gate(mat, qidxs, max_bond_dimension)
                self.current_state.normalise()

        self.output_state = self.current_state
        self.output_state.normalise()

        if samples:
            return self.output_state.sample_bitstrings(samples)
        return self.output_state

    def from_qiskit_gate(self, inst: CircuitInstruction) -> MatrixProductOperator:  # type: ignore
        return MatrixProductOperator.from_qiskit_gate(inst)

    def get_operator_mpo(
        self,
        after_gate: int | None = None,
        max_bond: int | None = None,
    ) -> MatrixProductOperator:
        qc_data = (
            self.circuit.data if after_gate is None else self.circuit.data[:after_gate]
        )
        qc = QuantumCircuit(self.num_qubits)
        for inst in qc_data:
            qc.append(inst)
        return MatrixProductOperator.from_qiskit_circuit(qc, max_bond=max_bond)
