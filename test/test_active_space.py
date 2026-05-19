from multiprocessing.util import info
import os

import pytest
import pyscf
from pyscf.mcscf import avas
import json

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


#-------------------------------------------------------------------------------

# Run HF:
def run_hf(molecule_file: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    location = os.path.join(repo_root, "molecules/mol_info", molecule_file)
    with open(location, "r") as f:
        info = json.load(f)
        mol_string = info["xyz"]
        mol_basis = "3-21G"
        mol_charge = info["charge"]

    mol_obj = pyscf.M(
            atom=mol_string,
            basis=mol_basis,
            charge=mol_charge,
        )
    rhf_obj = pyscf.scf.RHF(mol_obj).run()
    return rhf_obj

# Run CASCI, pick the number of active orbitals but let CASCI pick which ones
def run_casci_auto(rhf_obj):
    norb = 4  
    nelec = 4  
    casci_obj = pyscf.mcscf.CASCI(rhf_obj, norb, nelec)
    casci_result = casci_obj.kernel()
    energy = casci_result[0]
    return energy

# Run CASCI using a built-in pyscf active space selection method (AVAS - not TN based)
def run_casci_avas(rhf_obj):
    ncas, nelecas, mo=avas.avas(
    rhf_obj,
    ['N 2p'],
    )
    mc=pyscf.mcscf.CASCI(rhf_obj, ncas, nelecas)
    casci_result = mc.kernel(mo)
    energy = casci_result[0]
    return energy

# Run CASCI with your selected orbitals
def run_casci_selected(rhf_obj, active_space):
    norb = len(active_space)
    nelec = int(rhf_obj.mo_occ[norb - 1].sum())
    casci_obj = pyscf.mcscf.CASCI(rhf_obj, norb, nelec)
    orbs_casci = casci_obj.sort_mo(active_space)
    casci_result = casci_obj.kernel(orbs_casci)
    energy = casci_result[0]
    return energy

rhf_obj = run_hf("N2.json")
energy_auto = run_casci_auto(rhf_obj)
energy_avas = run_casci_avas(rhf_obj)
energy_selected = run_casci_selected(rhf_obj, active_space=[0, 1, 2, 3])
print(f"Energy with auto-selected active space: {energy_auto}")
print(f"Energy with AVAS-selected active space: {energy_avas}")
print(f"Energy with your selected active space: {energy_selected}")