from dataclasses import dataclass
from enum import Enum

import numpy as np
from cached_property import cached_property
from openfermion import FermionOperator, InteractionOperator, count_qubits
from openfermion.ops.representations import get_active_space_integrals
from openfermion.transforms.opconversions.bravyi_kitaev import (
    _bravyi_kitaev_interaction_operator,
)
from openfermion.transforms.opconversions.jordan_wigner import (
    _jordan_wigner_interaction_op,
)
from pyscf import ao2mo
from qiskit.quantum_info import Pauli, SparsePauliOp


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


def array_to_dict_nonzero_indices(arr, tol=1e-10):
    """
    A helper function to build Hamiltonians.
    """
    where_nonzero = np.where(~np.isclose(arr, 0, atol=tol))
    nonzero_indices = list(zip(*where_nonzero))
    return dict(zip(nonzero_indices, arr[where_nonzero]))


class MolecularIntegrals:
    """
    A helper class to build Hamiltonians.
    """

    def __init__(self, scf_obj, index_ordering="physicist") -> None:
        self.scf_obj = scf_obj
        self.c_matrix = scf_obj.mo_coeff
        self.n_spatial_orbs = scf_obj.mol.nao
        self.index_ordering = index_ordering
        self.n_spin_orbs = self.n_qubits = 2 * self.n_spatial_orbs
        self.n_electrons = scf_obj.mol.nelectron
        self.fermionic_molecular_hamiltonian = None
        self.qubit_molecular_hamiltonian = None

    @cached_property
    def _one_body_integrals(self):
        one_body_integrals = self.c_matrix.T @ self.scf_obj.get_hcore() @ self.c_matrix
        return one_body_integrals

    @cached_property
    def _two_body_integrals(self):
        two_body_integrals = ao2mo.restore(
            1,
            ao2mo.kernel(self.scf_obj.mol, self.scf_obj.mo_coeff),
            self.scf_obj.mo_coeff.shape[1],
        )
        if self.index_ordering == "physicist":
            two_body_integrals = two_body_integrals.transpose(0, 2, 3, 1)
        return two_body_integrals

    @cached_property
    def core_h_spin_basis(self):
        h_core_mo_basis_spin = np.zeros([self.n_spin_orbs] * 2)
        h_core_mo_basis_spin[::2, ::2] = self._one_body_integrals
        h_core_mo_basis_spin[1::2, 1::2] = self._one_body_integrals
        return h_core_mo_basis_spin

    @cached_property
    def eri_spin_basis(self):
        # alpha/beta electron indexing for chemists' and physicists' notation
        if self.index_ordering == "chemist":
            a, b, c, d = (0, 0, 1, 1)
            e, f, g, h = (1, 1, 0, 0)
        elif self.index_ordering == "physicist":
            a, b, c, d = (0, 1, 1, 0)
            e, f, g, h = (1, 0, 0, 1)

        eri_mo_basis_spin = np.zeros([self.n_spin_orbs] * 4)
        # same spin (even *or* odd indices)
        eri_mo_basis_spin[::2, ::2, ::2, ::2] = self._two_body_integrals
        eri_mo_basis_spin[1::2, 1::2, 1::2, 1::2] = self._two_body_integrals
        # different spin (even *and* odd indices)
        eri_mo_basis_spin[a::2, b::2, c::2, d::2] = self._two_body_integrals
        eri_mo_basis_spin[e::2, f::2, g::2, h::2] = self._two_body_integrals
        return eri_mo_basis_spin

    def get_active_space_spin_ints(self, occupied_indices, active_indices):
        ### get active space in SPATIAL setting
        core_const_shift, one_body_spatial_ints, two_body_spatial_ints = (
            get_active_space_integrals(
                self._one_body_integrals,
                self._two_body_integrals,
                occupied_indices=occupied_indices,
                active_indices=active_indices,
            )
        )

        ### convert active space spatial integrals to SPIN ints!
        n_spin_orbs = 2 * len(active_indices)
        h_core_mo_basis_spin = np.zeros([n_spin_orbs] * 2)
        h_core_mo_basis_spin[::2, ::2] = one_body_spatial_ints
        h_core_mo_basis_spin[1::2, 1::2] = one_body_spatial_ints

        if self.index_ordering == "chemist":
            a, b, c, d = (0, 0, 1, 1)
            e, f, g, h = (1, 1, 0, 0)
        elif self.index_ordering == "physicist":
            a, b, c, d = (0, 1, 1, 0)
            e, f, g, h = (1, 0, 0, 1)

        eri_mo_basis_spin = np.zeros([n_spin_orbs] * 4)
        # same spin (even *or* odd indices)
        eri_mo_basis_spin[::2, ::2, ::2, ::2] = two_body_spatial_ints
        eri_mo_basis_spin[1::2, 1::2, 1::2, 1::2] = two_body_spatial_ints
        # different spin (even *and* odd indices)
        eri_mo_basis_spin[a::2, b::2, c::2, d::2] = two_body_spatial_ints
        eri_mo_basis_spin[e::2, f::2, g::2, h::2] = two_body_spatial_ints

        return core_const_shift, h_core_mo_basis_spin, eri_mo_basis_spin


def get_hamiltonian(
    scf_obj=None,
    constant_shift=0,
    hcore=None,
    eri=None,
    operator_type="qubit",
    qubit_transformation="JW",
):
    if scf_obj is not None:
        integral_storage = MolecularIntegrals(scf_obj)
        constant_shift = scf_obj.energy_nuc()
        hcore = integral_storage.core_h_spin_basis
        eri = integral_storage.eri_spin_basis
        n_qubits = integral_storage.n_qubits
    else:
        assert hcore is not None and eri is not None, "Must supply molecular integrals"
        n_qubits = eri.shape[0]

    if operator_type == "qubit":
        interaction_operator = InteractionOperator(
            constant=constant_shift, one_body_tensor=hcore, two_body_tensor=eri * 0.5
        )
        if qubit_transformation in ["JW", "jordan_wigner"]:
            qubit_op = _jordan_wigner_interaction_op(
                interaction_operator, n_qubits=n_qubits
            )
        elif qubit_transformation in ["BK", "bravyi_kitaev"]:
            qubit_op = _bravyi_kitaev_interaction_operator(
                interaction_operator, n_qubits=n_qubits
            )
        else:
            raise ValueError("Unrecognised qubit transformation")
        return qubit_op

    elif operator_type == "fermion":
        one_body_coefficients = array_to_dict_nonzero_indices(hcore)
        two_body_coefficients = array_to_dict_nonzero_indices(eri)

        fermionic_molecular_hamiltonian = FermionOperator("", constant_shift)
        for (p, q), coeff in one_body_coefficients.items():
            fermionic_molecular_hamiltonian += FermionOperator(f"{p}^ {q}", coeff)
        for (p, q, r, s), coeff in two_body_coefficients.items():
            fermionic_molecular_hamiltonian += FermionOperator(
                f"{p}^ {q}^ {r} {s}", coeff * 0.5
            )
        return fermionic_molecular_hamiltonian


def qubitop_to_pauliop(ham):
    """
    Convert OpenFermion QubitOp to Qiskit SparsePauliOp.
    """
    n_qubits = count_qubits(ham)
    qubit_operator = ham
    paulis = []
    coeffs = []

    for qubit_terms, coefficient in qubit_operator.terms.items():
        pauli_label = ["I" for _ in range(n_qubits)]
        coeff = coefficient

        for tensor_term in qubit_terms:
            pauli_label[tensor_term[0]] = tensor_term[1]

        pauli_label = "".join(pauli_label)

        paulis.append(Pauli(pauli_label))
        coeffs.append(coeff)

    pauliOp = SparsePauliOp(paulis, coeffs=coeffs)

    return pauliOp


def pauliop_to_dict(ham):
    """
    Convert a Qiskit SparsePauliOp to a dictionary.
    """
    ham_dict = {str(ham[i].paulis[0]): ham[i].coeffs[0].real for i in range(len(ham))}
    return ham_dict


def ham_dict_from_scf(scf_obj, qubit_transformation="JW"):
    """
    Build the Hamiltonian dictionary directly from the scf object.
    """
    ham = get_hamiltonian(scf_obj, qubit_transformation=qubit_transformation)
    qiskit_ham = qubitop_to_pauliop(ham)
    ham_dict = pauliop_to_dict(qiskit_ham)
    return ham_dict
