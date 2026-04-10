"""
Module 2: Pauli Twirling Transpiler
Transpiles a Qiskit circuit to a target backend and applies Pauli twirling
to all two-qubit gates, converting coherent errors into stochastic Pauli noise.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import CXGate, CZGate, ECRGate, UnitaryGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers import BackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ---------------------------------------------------------------------------
# Pauli matrices (used to build twirl sets)
# ---------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

_PAULIS_1Q: dict[str, np.ndarray] = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}

# All 16 two-qubit Pauli tensor products P ⊗ Q
_PAULIS_2Q: list[tuple[str, str, np.ndarray]] = [
    (p, q, np.kron(qm, pm))  # q0=p, q1=q in circuit → kron(q1,q0)
    for p, pm in _PAULIS_1Q.items()
    for q, qm in _PAULIS_1Q.items()
]


def _build_twirl_set(
    gate_unitary: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Find all (P, Q) Pauli pairs such that

        (P⊗Q) · U · (P⊗Q)† = e^{iφ} U

    i.e. the Pauli pair commutes with U up to global phase.
    Returns list of (left_A, left_B, right_A, right_B) single-qubit gates
    where the twirl is  (A⊗B) U (A⊗B)†.
    Because we want the frame to cancel, the right gates equal the left gates
    (the correction is just the same Pauli applied again on the output side).

    For a gate G the twirled channel is:
        T(G) = (1/|S|) Σ_{P∈S} (P⊗Q) G (P⊗Q)†
    """
    U = gate_unitary
    valid = []
    for p_str, q_str, PQ in _PAULIS_2Q:
        # Correct twirl condition: PQ U PQ = e^{iφ} U
        # (PQ is self-inverse so PQ† = PQ)
        dressed = PQ @ U @ PQ
        # Check dressed = e^{iφ} U by verifying dressed @ U† = e^{iφ} I
        overlap = dressed @ U.conj().T
        # overlap should be a scalar multiple of identity
        # extract candidate phase from (0,0) element
        phase = overlap[0, 0]
        if abs(phase) < 1e-8:
            continue
        if not np.isclose(abs(phase), 1.0, atol=1e-8):
            continue
        if not np.allclose(overlap, phase * np.eye(4), atol=1e-8):
            continue
        PA = _PAULIS_1Q[p_str]
        QB = _PAULIS_1Q[q_str]
        # Right corrections: same Paulis (self-inverse), global phase is
        # unobservable so we don't need to track e^{iφ}
        valid.append((PA, QB, PA, QB))
    return valid


# Pre-compute twirl sets for common gates
_CNOT_U = Operator(CXGate()).data
_CZ_U = Operator(CZGate()).data
_ECR_U = Operator(ECRGate()).data

_TWIRL_SETS: dict[str, list] = {
    "cx": _build_twirl_set(_CNOT_U),
    "cz": _build_twirl_set(_CZ_U),
    "ecr": _build_twirl_set(_ECR_U),
}


def _pauli_to_gate(matrix: np.ndarray, label: str) -> Gate:
    """Wrap a 2x2 Pauli matrix as a 1-qubit Qiskit Gate."""
    return UnitaryGate(matrix, label=label)


class PauliTwirlingTranspiler:
    """
    Transpile a QuantumCircuit to a target backend and apply Pauli twirling
    to every two-qubit gate in the transpiled circuit.

    Pauli twirling wraps each two-qubit gate G with a randomly sampled
    Pauli pair (P, Q) from the gate's twirl set:

        G  →  (P⊗Q) · G · (P⊗Q)†

    The ideal action of G is preserved on average; coherent errors in G
    are converted into a stochastic Pauli channel (depolarising-like),
    which is both more benign and easier to characterise/mitigate.

    Parameters
    ----------
    backend : BackendV2
        Target backend.
    num_twirl_circuits : int
        Number of twirled circuit instances to generate per input circuit.
        Results should be averaged over all instances.
    optimization_level : int
        Qiskit transpiler optimisation level (0–3).
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        backend: BackendV2,
        num_twirl_circuits: int = 16,
        optimization_level: int = 1,
        seed: int | None = None,
    ) -> None:
        self.backend = backend
        self.num_twirl_circuits = num_twirl_circuits
        self.optimization_level = optimization_level
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transpile(self, circuit: QuantumCircuit) -> list[QuantumCircuit]:
        """
        Transpile *circuit* and return ``num_twirl_circuits`` independently
        twirled copies.  Average measurement results over all copies to
        obtain the twirled expectation value.

        Returns
        -------
        list[QuantumCircuit]
            Length = ``self.num_twirl_circuits``.
        """
        transpiled = self._base_transpile(circuit)
        return [
            self._apply_twirling(transpiled) for _ in range(self.num_twirl_circuits)
        ]

    def transpile_single(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Return a single randomly twirled transpiled circuit."""
        return self._apply_twirling(self._base_transpile(circuit))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _base_transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level,
            basis_gates=self.backend.basis_gates,
            coupling_map=self.backend.coupling_map,
        )
        return pm.run(circuit)

    def _apply_twirling(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Insert random Pauli twirls around every supported 2Q gate."""
        dag = circuit_to_dag(circuit)
        new_dag = DAGCircuit()

        # Copy registers
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        for node in dag.topological_op_nodes():
            gate_name = node.op.name.lower()
            if gate_name in _TWIRL_SETS and len(node.qargs) == 2:
                twirl_set = _TWIRL_SETS[gate_name]
                if len(twirl_set) == 0:
                    new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                    continue

                idx = self.rng.integers(len(twirl_set))
                P_left, Q_left, P_right, Q_right = twirl_set[idx]
                q0, q1 = node.qargs

                # Determine Pauli labels for cleaner circuit diagrams
                p_label = self._matrix_to_label(P_left)
                q_label = self._matrix_to_label(Q_left)

                # Left twirl gates
                new_dag.apply_operation_back(_pauli_to_gate(P_left, p_label), [q0], [])
                new_dag.apply_operation_back(_pauli_to_gate(Q_left, q_label), [q1], [])
                # Original gate
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                # Right correction gates
                new_dag.apply_operation_back(_pauli_to_gate(P_right, p_label), [q0], [])
                new_dag.apply_operation_back(_pauli_to_gate(Q_right, q_label), [q1], [])
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return dag_to_circuit(new_dag)

    @staticmethod
    def _matrix_to_label(m: np.ndarray) -> str:
        for label, ref in _PAULIS_1Q.items():
            if np.allclose(m, ref):
                return label
        return "P"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def supported_gates(self) -> list[str]:
        """Gates that can be twirled."""
        return list(_TWIRL_SETS.keys())

    def twirl_set_size(self, gate_name: str) -> int:
        """Number of valid twirling Pauli pairs for a given gate."""
        key = gate_name.lower()
        if key not in _TWIRL_SETS:
            raise ValueError(f"Gate '{gate_name}' not in twirl set.")
        return len(_TWIRL_SETS[key])
