"""
Noisy Circuit Decomposition — Layered Superoperator Form
===================================================================

Takes a Qiskit circuit, transpiles it to a given backend with a given noise
model, and decomposes it into an ordered list of layers.  Each layer is a
dict entry:

    {
        "layer_1": [
            {
                "qubits":     (0,),
                "gate_label": "sx",
                "ideal_ptm":  np.ndarray (4×4 or 16×16),
                "noise_ptm":  np.ndarray,
                "noisy_ptm":  np.ndarray,  # noise_ptm @ ideal_ptm
            },
            ...
        ],
        "layer_2": [...],
    }

A "layer" groups gates that act on disjoint qubits and can therefore be
applied simultaneously.  The noisy evolution of the full circuit is:

    L_total = Π_{layers} Π_{gates in layer} (N_gate · U_gate)

All PTMs are in the Pauli Transfer Matrix (superoperator) representation.

Dependencies: qiskit, qiskit-aer, numpy, and the pauli_twirl_noise module
from this package.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.providers import BackendV2
from qiskit.quantum_info import Kraus, Operator
from qiskit_aer.noise import NoiseModel

from .pauli_twirling_noise import (
    _unitary_to_ptm,
)

# ---------------------------------------------------------------------------
# Helpers: extract PTM from a Qiskit noise error object
# ---------------------------------------------------------------------------


def _kraus_to_ptm(kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Convert a list of Kraus operators to a PTM (via SuperOp)."""
    n_qubits = int(np.log2(kraus_ops[0].shape[0]))
    n_basis = 4**n_qubits
    # Normalized single-qubit Pauli basis
    paulis_1q = [
        np.eye(2) / np.sqrt(2),
        np.array([[0, 1], [1, 0]]) / np.sqrt(2),
        np.array([[0, -1j], [1j, 0]]) / np.sqrt(2),
        np.array([[1, 0], [0, -1]]) / np.sqrt(2),
    ]

    # Build n-qubit basis via tensor products
    if n_qubits == 1:
        basis = paulis_1q
    else:
        basis = [np.kron(b1, b2) for b1 in paulis_1q for b2 in paulis_1q]

    ptm = np.zeros((n_basis, n_basis), dtype=float)
    for j, Pj in enumerate(basis):
        evolved = sum(K @ Pj @ K.conj().T for K in kraus_ops)
        for i, Pi in enumerate(basis):
            ptm[i, j] = np.real(np.trace(Pi @ evolved)) / (2**n_qubits)
    return ptm


def _noise_error_to_ptm(error, n_qubits: int) -> np.ndarray:
    """
    Convert a Qiskit AER QuantumError to a PTM.
    Handles Kraus, unitary, and Pauli error types.
    """
    try:
        kraus = Kraus(error.to_quantumchannel())
        ops = [np.array(K) for K in kraus.data]
        return _kraus_to_ptm(ops)
    except Exception:
        pass

    # Fallback: identity PTM
    dim = 4**n_qubits
    return np.eye(dim)


def _identity_ptm(n_qubits: int) -> np.ndarray:
    dim = 4**n_qubits
    return np.eye(dim)


def _normalised_pauli_basis(n_qubits: int) -> list[np.ndarray]:
    """
    Returns the n-qubit normalised Pauli basis as a list of (2^n, 2^n) matrices,
    ordered by tensor product: II, IX, IY, IZ, XI, XX, ...
    Each matrix P satisfies Tr(P†P) = 1.
    """
    P1 = [
        np.eye(2) / np.sqrt(2),
        np.array([[0, 1], [1, 0]]) / np.sqrt(2),
        np.array([[0, -1j], [1j, 0]]) / np.sqrt(2),
        np.array([[1, 0], [0, -1]]) / np.sqrt(2),
    ]
    basis = [np.array([[1.0]])]
    for _ in range(n_qubits):
        basis = [np.kron(b, p) for b in basis for p in P1]
    return basis


def ptm_to_liouville(ptm: np.ndarray) -> np.ndarray:
    """
    Convert a Pauli Transfer Matrix to the Liouville (vectorised superoperator)
    representation.

    The PTM acts on the normalised Pauli basis:
        rho_pauli_out[i] = sum_j PTM[i,j] * rho_pauli_in[j]

    The Liouville superoperator acts on the column-vectorised density matrix:
        vec(rho)_out = M_liouville @ vec(rho)_in

    where vec(rho) stacks columns of rho: vec(rho)[i*d + j] = rho[j, i]
    (NumPy column-major / Fortran order).

    The two are related by the change-of-basis matrix U whose rows are the
    normalised Pauli basis elements laid out as computational-basis vectors:
        U[k, :] = vec(P_k)†   →   M_liouville = U† @ PTM @ U

    Parameters
    ----------
    ptm : np.ndarray, shape (4^n, 4^n)
        Pauli Transfer Matrix for an n-qubit channel. Accepts n=1 (4x4)
        and n=2 (16x16), and arbitrary n generally.

    Returns
    -------
    liouville : np.ndarray, shape (4^n, 4^n), dtype complex
        Liouville superoperator in the computational basis.
    """
    dim_liouville = ptm.shape[0]
    n_qubits = round(np.log2(np.sqrt(dim_liouville)))

    assert ptm.shape == (dim_liouville, dim_liouville), "PTM must be square"
    assert 4**n_qubits == dim_liouville, "PTM dimension must be 4^n"

    # Build the change-of-basis matrix U, shape (4^n, 4^n)
    # U[k, :] = vec(P_k) where vec stacks columns (Fortran order)
    basis = _normalised_pauli_basis(n_qubits)
    U = np.array(
        [
            P.flatten(order="F")  # column-stack: vec(P)[i*d+j] = P[j,i]
            for P in basis
        ],
        dtype=complex,
    )  # shape (4^n, 4^n), rows are vec(P_k)†...
    # actually rows are vec(P_k), so U† has them as cols

    # M_liouville = U† PTM U
    # Derivation: rho = sum_k c_k P_k, so vec(rho) = U† c
    #             c_out = PTM c_in  →  vec(rho_out) = U† PTM U vec(rho_in)
    liouville = U.conj().T @ ptm @ U

    return liouville


def liouville_to_ptm(liouville: np.ndarray) -> np.ndarray:
    """Inverse of ptm_to_liouville."""
    dim = liouville.shape[0]
    n_qubits = round(np.log2(np.sqrt(dim)))
    basis = _normalised_pauli_basis(n_qubits)
    U = np.array([P.flatten(order="F") for P in basis], dtype=complex)
    return np.real(U @ liouville @ U.conj().T)


# ---------------------------------------------------------------------------
# Layer extraction from a transpiled circuit
# ---------------------------------------------------------------------------


def _circuit_to_layers(circuit: QuantumCircuit) -> list[list[DAGOpNode]]:  # type: ignore
    """
    Convert a circuit to a list of layers (greedy left-to-right grouping).
    Gates within the same layer act on disjoint qubit sets.
    """
    dag = circuit_to_dag(circuit)
    layers: list[list[DAGOpNode]] = []  # type: ignore
    current_layer: list[DAGOpNode] = []  # type: ignore
    used_qubits: set = set()

    for node in dag.topological_op_nodes():
        if node.op.name in ("barrier", "delay"):
            continue
        node_qubits = {circuit.find_bit(q).index for q in node.qargs}
        if node_qubits & used_qubits:
            # Flush current layer
            if current_layer:
                layers.append(current_layer)
            current_layer = [node]
            used_qubits = node_qubits
        else:
            current_layer.append(node)
            used_qubits |= node_qubits

    if current_layer:
        layers.append(current_layer)

    return layers


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class NoisyCircuitDecomposer:
    """
    Transpile a circuit to a backend, attach a noise model, and decompose
    the result into a layer-by-layer superoperator description.

    Each gate in each layer is represented as:
        - ``ideal_ptm``  : PTM of the ideal unitary
        - ``noise_ptm``  : PTM of the error channel attached to the gate
        - ``noisy_ptm``  : ``noise_ptm @ ideal_ptm``  (full noisy operation)

    The complete circuit superoperator is:

        L = Π_layers  Π_{gates in layer}  noisy_ptm_gate

    Parameters
    ----------
    backend : BackendV2
        Transpilation target.
    noise_model : NoiseModel | None
        Qiskit AER noise model.  If None, all noise PTMs are identities.
    optimization_level : int
        Qiskit transpiler optimisation level.
    """

    def __init__(
        self,
        backend: BackendV2,
        noise_model: NoiseModel | None = None,
        optimization_level: int = 1,
    ) -> None:
        self.backend = backend
        self.noise_model = noise_model
        self.optimization_level = optimization_level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, circuit: QuantumCircuit) -> dict[str, list[dict[str, Any]]]:
        """
        Transpile *circuit* and return the layered superoperator decomposition.

        Returns
        -------
        dict
            Keys ``"layer_1"``, ``"layer_2"``, ... each mapping to a list of
            gate dicts with keys:
            ``qubits``, ``gate_label``, ``ideal_ptm``, ``noise_ptm``,
            ``noisy_ptm``.
        """
        transpiled = transpile(
            circuit,
            backend=self.backend,
            optimization_level=self.optimization_level,
        )
        layers = _circuit_to_layers(transpiled)
        result: dict[str, list[dict]] = {}

        for layer_idx, layer_nodes in enumerate(layers):
            layer_key = f"layer_{layer_idx + 1}"
            layer_ops: list[dict] = []

            for node in layer_nodes:
                gate_info = self._process_node(node, transpiled)
                if gate_info is not None:
                    layer_ops.append(gate_info)

            if layer_ops:
                result[layer_key] = layer_ops

        return result

    def full_circuit_ptm(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        Return the full-circuit superoperator as a single PTM
        (product of all layer PTMs, assuming independent single-qubit spaces).

        Note: this is the *per-gate* PTM product — for multi-qubit circuits
        with entangling gates, use the layered representation directly and
        tensor-compose gate PTMs appropriately for your specific qubit layout.
        """
        decomp = self.decompose(circuit)
        # Compose all noisy_ptm matrices in order
        total = None
        for layer_key in sorted(decomp.keys(), key=lambda k: int(k.split("_")[1])):
            for gate_info in decomp[layer_key]:
                ptm = gate_info["noisy_ptm"]
                if total is None:
                    total = ptm
                else:
                    total = ptm @ total
        return total if total is not None else np.eye(4)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_node(
        self,
        node: DAGOpNode,  # type: ignore
        circuit: QuantumCircuit,  # type: ignore
    ) -> dict | None:
        """Build the superoperator dict for a single gate node."""
        gate = node.op
        gate_name = gate.name.lower()

        if gate_name in ("measure", "reset", "barrier", "delay", "snapshot"):
            return None

        qubits = tuple(circuit.find_bit(q).index for q in node.qargs)
        n_qubits = len(qubits)

        # Ideal gate PTM
        try:
            U = Operator(gate).data
            ideal_ptm = _unitary_to_ptm(U)
        except Exception:
            ideal_ptm = _identity_ptm(n_qubits)
            U = None

        # Noise PTM from noise model
        noise_ptm = self._get_noise_ptm(gate_name, qubits, n_qubits)

        # Convert to Liouvillian representation
        ideal_liouville = ptm_to_liouville(ideal_ptm)
        noise_liouville = ptm_to_liouville(noise_ptm)

        return {
            "qubits": qubits,
            "gate_label": gate_name,
            "ideal_liouville": ideal_liouville,
            "noise_liouville": noise_liouville,
            "noisy_liouville": noise_liouville @ ideal_liouville,
            "ideal_ptm": ideal_ptm,
            "noise_ptm": noise_ptm,
            "noisy_ptm": noise_ptm @ ideal_ptm,
        }

    def _get_noise_ptm(
        self, gate_name: str, qubits: tuple, n_qubits: int
    ) -> np.ndarray:
        """Extract the noise PTM for a given gate from the noise model."""
        if self.noise_model is None:
            return _identity_ptm(n_qubits)

        # Try to find a matching error in the noise model
        errors = self.noise_model._local_quantum_errors.get(gate_name, {})
        qubits_key = qubits if qubits in errors else tuple(reversed(qubits))

        if qubits_key in errors:
            error = errors[qubits_key]
            return _noise_error_to_ptm(error, n_qubits)

        # Fall back to gate-wide (non-local) errors
        gate_errors = self.noise_model._default_quantum_errors.get(gate_name, None)
        if gate_errors is not None:
            return _noise_error_to_ptm(gate_errors, n_qubits)

        return _identity_ptm(n_qubits)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_decomposition(decomp: dict[str, list[dict]]) -> None:
        """Pretty-print the layered decomposition (without full matrices)."""
        for layer_key in sorted(decomp.keys(), key=lambda k: int(k.split("_")[1])):
            print(f"\n{'='*50}")
            print(f"  {layer_key.upper()}")
            print(f"{'='*50}")
            for gate_info in decomp[layer_key]:
                qstr = ", ".join(str(q) for q in gate_info["qubits"])
                print(
                    f"  Gate: {gate_info['gate_label'].upper():8s}  "
                    f"qubits: ({qstr})"
                )
                n = gate_info["ideal_ptm"].shape[0]
                ideal_diag = np.diag(gate_info["ideal_ptm"])
                noise_diag = np.diag(gate_info["noise_ptm"])
                print(
                    f"    ideal_ptm  : {n}×{n}, "
                    f"diag range [{ideal_diag.min():.3f}, {ideal_diag.max():.3f}]"
                )
                print(
                    f"    noise_ptm  : {n}×{n}, "
                    f"diag range [{noise_diag.min():.3f}, {noise_diag.max():.3f}]"
                )
