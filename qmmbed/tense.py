import logging
import random
from dataclasses import dataclass
from enum import Enum

import quimb.tensor as qtn
import sparse
from openfermion.ops import FermionOperator, QubitOperator
from qiskit import QuantumCircuit
from quimb.tensor import DMRG2, MatrixProductState

logger = logging.getLogger(__name__)


class PauliTerm(Enum):
    I = "I"  # noqa: E741
    X = "X"  # noqa: E741
    Y = "Y"  # noqa: E741
    Z = "Z"  # noqa: E741


@dataclass(frozen=True)
class UpdateValues:
    indices: tuple[int, int, int, int]
    weights: tuple[complex, complex]


def qiskit_to_quimb_circuit(qiskit_circ: QuantumCircuit) -> qtn.Circuit:
    """Transform a Qiskit quantum circuit to a Quimb quantum circuit.

    Args:
        qiskit_circ (QuantumCircuit): A Qiskit quantum circuit.

    Returns:
        qtn.Circuit: A Quimb quantum circuit.

    Raises:
        ValueError: If the Qiskit gate is not supported in Quimb.
    """
    logger.debug("Transforming Qiskit circuit to Quimb circuit.")
    num_qubits = qiskit_circ.num_qubits
    quimb_circ = qtn.Circuit(N=num_qubits)

    for inst in qiskit_circ.data:
        quimb_gate_error = False
        inst_name = inst[0].name
        inst_params = inst[0].params if len(inst[0].params) > 0 else None

        if inst_name == "barrier":
            continue

        if inst_name.upper() == "U":
            inst_name = "U3"
        inst_qubits = [inst.qubits.index(q) for q in inst.qubits]

        if inst_params is not None:
            if inst_name.upper() not in qtn.circuit.ALL_PARAM_GATES:
                quimb_gate_error = True
            else:
                new_params = [
                    (
                        inst_params[i]
                        if isinstance(inst_params[i], float)
                        else random.random()
                    )
                    for i in range(len(inst_params))
                ]
                quimb_circ.apply_gate(inst_name.upper(), *new_params, *inst_qubits)
        else:
            if inst_name.upper() not in qtn.circuit.ALL_GATES:
                quimb_gate_error = True

        if quimb_gate_error:
            logger.error("Qiskit param gate {inst_name} is not supported in Quimb.")
            raise ValueError("Qiskit param gate {inst_name} is not supported in Quimb.")

        quimb_circ.apply_gate(inst_name.upper(), *inst_qubits)

    return quimb_circ


def _update_array(coords, data, weight, p_string_idx, pauli, offset=False):
    """
    Helper function to update array coordinates and data for the MPO tensors.
    """
    pauli_map = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}  # Map Pauli terms to indices
    if pauli not in pauli_map:
        raise ValueError(f"Invalid Pauli term: {pauli}")

    pauli_idx = pauli_map[pauli]
    if offset:
        coords[0].append(p_string_idx)
        coords[1].append(p_string_idx)
        coords[2].append(pauli_idx)
        coords[3].append(pauli_idx)
        data.append(weight)
    else:
        coords[0].append(p_string_idx)
        coords[1].append(pauli_idx)
        coords[2].append(pauli_idx)
        data.append(weight)


def _hamiltonian_to_mpo(
        hamiltonian: dict,
) -> qtn.MatrixProductOperator:
    num_qubits = len(list(hamiltonian.keys())[0])  # Number of qubits
    num_ham_terms = len(hamiltonian.keys())  # Number of Hamiltonian terms

    # Initialise arrays for first, middle, and last MPO tensors
    first_array_coords = [[], [], []]
    middle_array_coords = [[[], [], [], []] for _ in range(1, num_qubits - 1)]
    last_array_coords = [[], [], []]
    first_array_data = []
    middle_array_data = [[] for _ in range(1, num_qubits - 1)]
    last_array_data = []

    for p_string_idx, (p_string, weights) in enumerate(hamiltonian.items()):
        weight = weights[0]  # Extract the weight
        p_string = p_string[::-1]  # Reverse the Pauli string (qubit indexing convention)

        # First Term
        _update_array(
            first_array_coords, first_array_data, weight, p_string_idx, p_string[0]
        )

        # Middle Terms
        for p_idx in range(1, num_qubits - 1):
            _update_array(
                middle_array_coords[p_idx - 1],
                middle_array_data[p_idx - 1],
                1.0,  # Middle terms weight is neutral
                p_string_idx,
                p_string[p_idx],
                offset=True,
            )

        # Final Term
        _update_array(
            last_array_coords, last_array_data, 1.0, p_string_idx, p_string[-1]
        )

    # Create sparse tensor arrays for the MPO
    first_array = sparse.COO(
        first_array_coords, first_array_data, shape=(num_ham_terms, 2, 2)
    )
    middle_arrays = [
        sparse.COO(
            middle_array_coords[i - 1],
            middle_array_data[i - 1],
            shape=(num_ham_terms, num_ham_terms, 2, 2),
        )
        for i in range(1, num_qubits - 1)
    ]
    last_array = sparse.COO(
        last_array_coords, last_array_data, shape=(num_ham_terms, 2, 2)
    )

    # Combine the tensors into a MatrixProductOperator
    return qtn.MatrixProductOperator([first_array] + middle_arrays + [last_array])


def get_dmrg_solution(
        ham: FermionOperator | QubitOperator, max_bond: int
) -> tuple[MatrixProductState, float]:
    logger.debug("Solving dmrg with max bond dimension %d.", max_bond)
    ham_mpo = _hamiltonian_to_mpo(ham)
    mydmrg = DMRG2(ham_mpo, bond_dims=[max_bond], cutoffs=1e-10)
    mydmrg.solve(tol=1e-6, verbosity=0)
    logger.debug("DMRG energy %d.", mydmrg.energy)
    return mydmrg.state, mydmrg.energy


def _update_array(
        array: list,
        data: list,
        weight: complex,
        p_string_idx: int,
        term: str,
        offset: bool = False,
) -> None:
    logger.debug("Updating array with term %d. offset=%d", term, offset)
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
