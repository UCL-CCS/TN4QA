import json
from dataclasses import dataclass
from enum import Enum
from functools import reduce

import ferrmion as fr
import numpy as np
import sparse
from pyscf import ao2mo


class PauliTerm(Enum):
    """
    A class to conveniently access Pauli operators.
    """

    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"


@dataclass(frozen=True)
class UpdateValues:
    """
    A helper class for buiding Hamiltonian MPOs.
    """

    indices: tuple[int, int, int, int]
    weights: tuple[complex, complex]


def _update_array(
    array: list,
    data: list,
    weight: complex,
    p_string_idx: int,
    term: str,
    offset: bool = False,
) -> None:
    """
    A helper function to build Hamiltonian MPOs.
    """
    match term:
        case PauliTerm.I.value:
            update_values = UpdateValues((0, 0, 1, 1), (1, 1))
        case PauliTerm.X.value:
            update_values = UpdateValues((0, 1, 1, 0), (1, 1))
        case PauliTerm.Y.value:
            update_values = UpdateValues((0, 1, 1, 0), (-1j, 1j))
        case PauliTerm.Z.value:
            update_values = UpdateValues((0, 0, 1, 1), (1, -1))

    for i in [0, 1]:
        array[0].append(p_string_idx)
        if offset:
            array[1].append(p_string_idx)

        array[1 + int(offset)].append(update_values.indices[2 * i])
        array[2 + int(offset)].append(update_values.indices[(2 * i) + 1])
        data.append(update_values.weights[i] * weight)


@dataclass(frozen=True)
class UpdateValuesFermion:
    """
    A helper class for buiding Hamiltonian MPOs.
    """

    indices: tuple[int, int]
    weights: tuple[complex]


def _update_array_fermion(
    array: list,
    data: list,
    weight: complex,
    string_idx: int,
    term: str,
    offset: bool = False,
) -> None:
    """
    A helper function to build Hamiltonian MPOs.
    """

    I = np.array([[1, 0], [0, 1]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    p = np.array([[0, 0], [1, 0]], dtype=complex)
    m = np.array([[0, 1], [0, 0]], dtype=complex)

    total_op = I.copy()
    for x in term:
        if x == "Z":
            total_op = total_op @ Z
        if x == "+":
            total_op = total_op @ p
        if x == "-":
            total_op = total_op @ m

    non_zero_vals = []
    for row in [0, 1]:
        for col in [0, 1]:
            if total_op[row, col] == 0.0:
                continue
            non_zero_vals.append((row, col, total_op[row, col]))

    if len(non_zero_vals) == 1:
        update_values = UpdateValuesFermion(
            (non_zero_vals[0][0], non_zero_vals[0][1]), (non_zero_vals[0][2],)
        )
    elif len(non_zero_vals) == 2:
        update_values = UpdateValues(
            (
                non_zero_vals[0][0],
                non_zero_vals[0][1],
                non_zero_vals[1][0],
                non_zero_vals[1][1],
            ),
            (non_zero_vals[0][2], non_zero_vals[1][2]),
        )

    for i in range(len(non_zero_vals)):
        array[0].append(string_idx)
        if offset:
            array[1].append(string_idx)

        array[1 + int(offset)].append(update_values.indices[2 * i])
        array[2 + int(offset)].append(update_values.indices[(2 * i) + 1])
        data.append(update_values.weights[i] * weight)


def array_to_dict_nonzero_indices(arr, tol=1e-10):
    """
    A helper function to build Hamiltonians.
    """
    where_nonzero = np.where(~np.isclose(arr, 0, atol=tol))
    nonzero_indices = list(zip(*where_nonzero))
    return dict(zip(nonzero_indices, arr[where_nonzero]))


class ReadMoleculeData:
    """
    A class to read information from molecule json files.
    """

    def __init__(self, filename):
        with open(filename) as f:
            data = json.load(f)

        self.geometry = data["geometry"]
        self.basis = data["basis"]
        self.num_electrons = data["electrons"]
        self.num_spatial_orbs = data["spatial_orbs"]
        self.num_spin_orbs = 2 * self.num_spatial_orbs

        self.rhf_energy = data["E_RHF"]
        self.ccsd_energy = data["E_CCSD"]
        self.ccsdpt_energy = data["E_CCSDpT"]
        self.fci_energy = data["E_FCI"]

        fci_sparse_vec = data["FCI_vector"]
        self.fci_vector = [0] * (2**self.num_spin_orbs)
        for idx, val in fci_sparse_vec.items():
            self.fci_vector[int(idx)] = val[0] + 1j * val[1]
        self.fci_vector = np.array(self.fci_vector)
        self.fci_vector_sparse = sparse.COO.from_numpy(self.fci_vector)

        self.qubit_hamiltonian = data["qubit_hamiltonian"]
        self.qubit_hamiltonian = {
            k: v[0] + 1j * v[1] for k, v in self.qubit_hamiltonian.items()
        }

        fermionic_hamiltonian = data["fermionic_hamiltonian"]
        self.one_electron_integrals = np.array(fermionic_hamiltonian[0])
        self.two_electron_integrals = np.array(fermionic_hamiltonian[1])
        self.nuclear_energy = fermionic_hamiltonian[2]
        self.fermionic_hamiltonian = (
            self.one_electron_integrals,
            self.two_electron_integrals,
            self.nuclear_energy,
        )


def stitch_matrices(A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
    """
    Stitch one or two N x N matrices to produce an N x 2N matrix with alternating columns.

    If only A is provided, it is duplicated and interleaved with itself.
    If both A and B are provided, they must be N x N and of the same shape.

    Args:
        A (np.ndarray): First N x N matrix.
        B (np.ndarray, optional): Second N x N matrix. Defaults to None.

    Returns:
        np.ndarray: N x 2N matrix with alternating columns from A and B (or A and A).
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be an N x N square matrix.")

    if B is None:
        B = A.copy()
    elif B.shape != A.shape:
        raise ValueError("Both matrices must be N x N and the same shape.")

    N = A.shape[0]
    C = np.empty((N, 2 * N), dtype=A.dtype)
    C[:, ::2] = A
    C[:, 1::2] = B
    return C


def split_matrices(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split an N x 2N matrix into two N x N matrices with alternating columns.

    Args:
        C (np.ndarray): An N x 2N matrix where columns alternate from A and B.

    Returns:
        tuple: Two N x N matrices (A, B) extracted from C.
    """
    if C.ndim != 2 or C.shape[1] % 2 != 0:
        raise ValueError("Input must be a 2D matrix with an even number of columns.")

    A = C[:, ::2]
    B = C[:, 1::2]
    return A, B


def spinorb_from_spatial_rhf(one_body, two_body):
    n_spatial = one_body.shape[0]
    n_qubits = 2 * n_spatial

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_spatial):
        for q in range(n_spatial):
            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body[p, q]
            one_body_coefficients[2 * p + 1, 2 * q + 1] = one_body[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_spatial):
                for s in range(n_spatial):
                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                        two_body[p, q, r, s]
                    )
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                        two_body[p, q, r, s]
                    )
                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = two_body[
                        p, q, r, s
                    ]
                    two_body_coefficients[
                        2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                    ] = two_body[p, q, r, s]

    # Truncate.
    # one_body_coefficients[np.absolute(one_body_coefficients) < 1e-8] = 0.0
    # two_body_coefficients[np.absolute(two_body_coefficients) < 1e-8] = 0.0

    return one_body_coefficients, two_body_coefficients


def spinorb_from_spatial_uhf(
    one_body_a, one_body_b, two_body_aa, two_body_bb, two_body_ab
):
    n_spatial = one_body_a.shape[0]
    n_qubits = 2 * n_spatial

    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    for p in range(n_spatial):
        for q in range(n_spatial):
            one_body_coefficients[2 * p, 2 * q] = one_body_a[p, q]  # α-α
            one_body_coefficients[2 * p + 1, 2 * q + 1] = one_body_b[p, q]  # β-β

            for r in range(n_spatial):
                for s in range(n_spatial):
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = two_body_aa[
                        p, q, r, s
                    ]
                    two_body_coefficients[
                        2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                    ] = two_body_bb[p, q, r, s]
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                        two_body_ab[p, q, r, s]
                    )
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                        two_body_ab[q, p, s, r]
                    )

    # one_body_coefficients[np.abs(one_body_coefficients) < 1e-8] = 0.0
    # two_body_coefficients[np.abs(two_body_coefficients) < 1e-8] = 0.0

    return one_body_coefficients, two_body_coefficients


def build_qham(scf_obj):
    C = scf_obj.mo_coeff
    S = scf_obj.get_ovlp()
    hcore = scf_obj.get_hcore()
    mol_obj = scf_obj.mol

    if len(C.shape) == 2:
        # scf_type = "RHF"
        n_orbitals = C.shape[1]
        assert np.allclose(
            C.T @ S @ C, np.eye(n_orbitals)
        ), "MO coefficients are not orthonormal"
        one_spatial = (
            reduce(np.dot, (C.T, hcore, C))
            .reshape(n_orbitals, n_orbitals)
            .astype(float)
        )
        two_spatial = ao2mo.kernel(mol_obj, C)
        two_spatial = ao2mo.restore(1, two_spatial, n_orbitals)
        two_spatial = np.asarray(two_spatial.transpose(0, 2, 3, 1), order="C")
        ones_spin, twos_spin = spinorb_from_spatial_rhf(one_spatial, two_spatial)
        twos_spin *= 0.5
    elif len(C.shape) == 3:
        # scf_type = "UHF"
        assert np.allclose(
            C[0].T @ S @ C[0], np.eye(C[0].shape[1])
        ), "MO coefficients are not orthonormal"
        assert np.allclose(
            C[1].T @ S @ C[1], np.eye(C[1].shape[1])
        ), "MO coefficients are not orthonormal"
        C_a = C[0]
        C_b = C[1]
        n_orbitals = C_a.shape[1]
        one_spatial_a = (
            reduce(np.dot, (C_a.T, hcore, C_a))
            .reshape(n_orbitals, n_orbitals)
            .astype(float)
        )
        one_spatial_b = (
            reduce(np.dot, (C_b.T, hcore, C_b))
            .reshape(n_orbitals, n_orbitals)
            .astype(float)
        )
        two_spatial_aa = ao2mo.kernel(mol_obj, C_a)
        two_spatial_aa = np.asarray(
            ao2mo.restore(1, two_spatial_aa, n_orbitals).transpose(0, 2, 3, 1),
            order="C",
        )
        two_spatial_bb = ao2mo.kernel(mol_obj, C_b)
        two_spatial_bb = np.asarray(
            ao2mo.restore(1, two_spatial_bb, n_orbitals).transpose(0, 2, 3, 1),
            order="C",
        )
        two_spatial_ab = ao2mo.kernel(mol_obj, (C_a, C_a, C_b, C_b))
        two_spatial_ab = np.asarray(
            ao2mo.restore(1, two_spatial_ab, n_orbitals).transpose(0, 2, 3, 1),
            order="C",
        )
        ones_spin, twos_spin = spinorb_from_spatial_uhf(
            one_spatial_a, one_spatial_b, two_spatial_aa, two_spatial_bb, two_spatial_ab
        )
        twos_spin *= 0.5
    else:
        raise ValueError("unknown SCF type")
    fham = fr.molecular_hamiltonian(
        one_e_coeffs=ones_spin,
        two_e_coeffs=twos_spin,
        constant_energy=mol_obj.energy_nuc(),
    )
    qham = fr.JordanWigner(fham.n_modes).encode(fham)
    return {
        k: v.real
        for k, v in sorted(qham.items(), key=lambda item: np.abs(item[1]), reverse=True)
    }
