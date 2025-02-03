from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict

import quimb.tensor as qtn
import sparse
import pyscf


class PauliTerm(Enum):
    """Enumeration of Pauli terms for quantum operators."""
    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"


@dataclass(frozen=True)
class UpdateValues:
    """Data class to hold indices and weights for MPO updates."""
    indices: Tuple[int, int, int, int]
    weights: Tuple[complex, complex]


def get_update_values(term: str) -> UpdateValues:
    """Retrieve the update values based on the Pauli term."""
    match term:
        case PauliTerm.I.value:
            return UpdateValues((0, 0, 1, 1), (1, 1))
        case PauliTerm.X.value:
            return UpdateValues((0, 1, 1, 0), (1, 1))
        case PauliTerm.Y.value:
            return UpdateValues((0, 1, 1, 0), (-1j, 1j))
        case PauliTerm.Z.value:
            return UpdateValues((0, 0, 1, 1), (1, -1))
        case _:
            raise ValueError(f"Invalid Pauli term: {term}")


def update_array(
        coords: List[List[int]],
        data: List[complex],
        weight: complex,
        term_idx: int,
        term: str,
        offset: bool = False,
) -> None:
    """
    Update coordinate and data arrays for the sparse tensor representation.

    Args:
        coords (List[List[int]]): The coordinate list to be updated.
        data (List[complex]): The data values to be updated.
        weight (complex): The weight of the term.
        term_idx (int): Index of the Pauli string term.
        term (str): The Pauli term ('I', 'X', 'Y', 'Z').
        offset (bool): Whether to apply an offset for middle terms.
    """
    update_values = get_update_values(term)
    for i in range(2):
        coords[0].append(term_idx)
        if offset:
            coords[1].append(term_idx)

        coords[1 + int(offset)].append(update_values.indices[2 * i])
        coords[2 + int(offset)].append(update_values.indices[2 * i + 1])
        data.append(update_values.weights[i] * weight)


def _hamiltonian_to_mpo(hamiltonian: Dict[str, Tuple[complex]]) -> qtn.MatrixProductOperator:
    """
    Convert a Hamiltonian dictionary to a Matrix Product Operator (MPO).

    Args:
        hamiltonian (Dict[str, Tuple[complex]]): Dictionary mapping Pauli strings to weights.

    Returns:
        qtn.MatrixProductOperator: The MPO representation of the Hamiltonian.
    """
    num_qubits = len(next(iter(hamiltonian)))  # Number of qubits
    num_terms = len(hamiltonian)  # Number of Hamiltonian terms

    # Initialize sparse array structures
    first_coords, first_data = [[] for _ in range(3)], []
    middle_coords = [[[] for _ in range(4)] for _ in range(num_qubits - 2)]
    middle_data = [[] for _ in range(num_qubits - 2)]
    last_coords, last_data = [[] for _ in range(3)], []

    for term_idx, (pauli_string, weights) in enumerate(hamiltonian.items()):
        weight = weights[0]  # Extract the weight
        pauli_string = pauli_string[::-1]  # Reverse the Pauli string for correct indexing

        # Update first term
        update_array(first_coords, first_data, weight, term_idx, pauli_string[0])

        # Update middle terms
        for qubit_idx in range(1, num_qubits - 1):
            update_array(
                middle_coords[qubit_idx - 1],
                middle_data[qubit_idx - 1],
                1.0,  # Middle terms have neutral weight
                term_idx,
                pauli_string[qubit_idx],
                offset=True,
            )

        # Update last term
        update_array(last_coords, last_data, 1.0, term_idx, pauli_string[-1])

    # Create sparse tensor arrays for MPO tensors
    first_tensor = sparse.COO(first_coords, first_data, shape=(num_terms, 2, 2))
    middle_tensors = [
        sparse.COO(coords, data, shape=(num_terms, num_terms, 2, 2))
        for coords, data in zip(middle_coords, middle_data)
    ]
    last_tensor = sparse.COO(last_coords, last_data, shape=(num_terms, 2, 2))

    # Combine tensors into a MatrixProductOperator
    return qtn.MatrixProductOperator([first_tensor] + middle_tensors + [last_tensor])

def load_scf_from_chk(
    chk_path : str,
    scf_method : str = 'RHF'
) -> pyscf.scf.hf.RHF:
    """
    Loads a PySCF SCF object from a .chk (checkpoint) file.

    Args:
        chk_path (str): path to the .chk file to load.
        scf_method (str): the SCF method to use (only RHF at the moment).

    Returns:
        scf_object (pyscf.scf.hf.RHF): the SCF object.
    """
    if scf_method == 'RHF':
        scf_object = pyscf.scf.RHF(
            mol = pyscf.scf.chkfile.load_scf(chk_path)[0]
        )
        scf_object.__dict__.update(pyscf.scf.chkfile.load(chk_path, 'scf'))
        scf_object.run()
        return scf_object
    else:
        raise NotImplementedError(f"SCF method {scf_method} not implemented.")
