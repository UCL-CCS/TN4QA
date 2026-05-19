import os

import pytest

from tn4qa.dmrg import DMRG
from tn4qa.mps import MatrixProductState
from tn4qa.tn_methods.active_space_selection import (
    autocas_selection_ranked_entropy_threshold,
    autocas_selection_ranked_fixed_number,
    autocas_selection_total_entropy,
)
from tn4qa.utils import ReadMoleculeData


MOLECULE_FILES = ["H2.json", "LiH.json", "N2.json"]


def build_mps_from_molecule(molecule_file: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    location = os.path.join(repo_root, "molecules", molecule_file)

    mol_data = ReadMoleculeData(location)
    ham = mol_data.qubit_hamiltonian
    hf_mps = MatrixProductState.from_hf_state(
        mol_data.num_spin_orbs, mol_data.num_electrons
    )

    dmrg = DMRG(hamiltonian=ham, max_mps_bond=2, initial_mps=hf_mps)
    _, mps = dmrg.run(5)

    assert isinstance(mps, MatrixProductState)
    return mol_data, mps


@pytest.mark.parametrize("molecule_file", MOLECULE_FILES)
def test_autocas_selection_ranked_fixed_number(molecule_file):
    mol_data, mps = build_mps_from_molecule(molecule_file)

    n_sites = mol_data.num_spin_orbs // 2
    active_orbitals = min(2, n_sites)

    top_orbitals = autocas_selection_ranked_fixed_number(
        mps, n_sites=n_sites, active_orbitals=active_orbitals
    )

    assert isinstance(top_orbitals, list)
    assert len(top_orbitals) == active_orbitals
    assert all(isinstance(idx, int) for idx in top_orbitals)
    assert all(0 <= idx < n_sites for idx in top_orbitals)


@pytest.mark.parametrize("molecule_file", MOLECULE_FILES)
def test_autocas_selection_total_entropy(molecule_file):
    mol_data, mps = build_mps_from_molecule(molecule_file)

    n_sites = mol_data.num_spin_orbs // 2
    threshold = 0.5

    top_orbitals = autocas_selection_total_entropy(
        mps, n_sites=n_sites, threshold=threshold
    )

    assert isinstance(top_orbitals, list)
    assert all(isinstance(idx, int) for idx in top_orbitals)
    assert all(0 <= idx < n_sites for idx in top_orbitals)


@pytest.mark.parametrize("molecule_file", MOLECULE_FILES)
def test_autocas_selection_ranked_entropy_threshold(molecule_file):
    mol_data, mps = build_mps_from_molecule(molecule_file)

    n_sites = mol_data.num_spin_orbs // 2

    top_orbitals = autocas_selection_ranked_entropy_threshold(
        mps, n_sites=n_sites
    )

    assert isinstance(top_orbitals, list)
    assert all(isinstance(idx, int) for idx in top_orbitals)
    assert all(0 <= idx < n_sites for idx in top_orbitals)