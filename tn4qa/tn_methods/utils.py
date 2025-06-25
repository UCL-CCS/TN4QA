import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from scipy.linalg import expm


def kak_recomposition(
    sq1_params: list[float],
    sq2_params: list[float],
    cartan_params: list[float],
    sq3_params: list[float],
    sq4_params: list[float],
) -> ndarray:
    """
    Given a set of parameters, form the corresponding 2-qubit unitary U = (sq1 x sq2) * entangling * (sq3 x sq4)

    Args:
        sq1_params: Three parameters defining a single qubit gate
        sq2_params: Three parameters defining a single qubit gate
        cartan_params: Three parameters defining an entangling gate
        sq3_params: Three parameters defining a single qubit gate
        sq4_params: Three parameters defining a single qubit gate

    Returns:
        The 4x4 matrix for U
    """
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])
    entangling_mat = expm(
        -1j
        * (
            cartan_params[0] * np.kron(x, x)
            + cartan_params[1] * np.kron(y, y)
            + cartan_params[2] * np.kron(z, z)
        )
    )
    entangling_gate = UnitaryGate(entangling_mat)

    qc = QuantumCircuit(2)
    qc.u(sq1_params[0], sq1_params[1], sq1_params[2], 0)
    qc.u(sq2_params[0], sq2_params[1], sq2_params[2], 1)
    qc.append(entangling_gate, [1, 0])
    qc.u(sq3_params[0], sq3_params[1], sq3_params[2], 0)
    qc.u(sq4_params[0], sq4_params[1], sq4_params[2], 1)

    op = Operator(qc).reverse_qargs().data
    return op


def single_qubit_gate(params: list[float]) -> ndarray:
    """
    Get the matrix of a generic 2 qubit gate

    Args:
        params: The three paramters controlling the gate

    Returns:
        The 2x2 matrix
    """
    qc = QuantumCircuit(1)
    qc.u(params[0], params[1], params[2], 0)
    op = Operator(qc).reverse_qargs().data
    return op


def givens_rotation(rot_angle: float) -> ndarray:
    """
    Build a Givens rotation

    Args:
        rot_angle: The rotation angle

    Returns:
        The 4x4 matrix
    """
    mat = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rot_angle), np.sin(rot_angle), 0],
            [0, -np.sin(rot_angle), np.cos(rot_angle), 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )
    return mat


def symmetry_preserving_two_qubit_gate(
    subspace1_param: float, subspace2_params: list[float], subspace3_param: float
) -> ndarray:
    """
    Build a gate that preserves number and S_z symmetry

    Args:
        subspace_1_param: The parameter that controls the phase on the |00> subspace
        subspace2_params: The three parameters defining the rotation on the |01>,|10> subspace
        subspace3_param: The parameter that controls the phase on the |11> subspace

    Returns:
        The 4x4 matrix
    """
    sq_gate = single_qubit_gate(subspace2_params)
    mat = np.array(
        [
            [np.exp(1j * subspace1_param), 0, 0, 0],
            [0, sq_gate[0, 0], sq_gate[0, 1], 0],
            [0, sq_gate[1, 0], sq_gate[1, 1], 0],
            [0, 0, 0, np.exp(1j * subspace3_param)],
        ]
    )
    return mat
