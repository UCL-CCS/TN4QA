import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
import sparse


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
