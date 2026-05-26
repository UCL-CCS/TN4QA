import json
import os

import numpy as np
import pyscf
import pytest
from pyscf.mcscf import avas

from tn4qa.dmrg import DMRG
from tn4qa.mps import MatrixProductState
from tn4qa.tn_methods.active_space_selection import (
    autocas_selection_ranked_entropy_threshold,
    autocas_selection_ranked_fixed_number,
    autocas_selection_total_entropy,
    ef_active_space_brute_force,
    ef_active_space_greedy,
    ef_active_space_sample,
)
from tn4qa.utils import ReadMoleculeData

MOLECULE_FILES = ["H2.json", "LiH.json", "N2.json"]
nactive = 6

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
    active_orbitals = min(nactive, n_sites)

    top_orbitals = autocas_selection_ranked_fixed_number(
        mps, n_sites=n_sites, active_orbitals=active_orbitals
    )

    print(f"Selected top {active_orbitals} orbitals for {molecule_file}: {top_orbitals} using AUTOCAS ranked fixed number method")
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

    top_orbitals = autocas_selection_ranked_entropy_threshold(mps, n_sites=n_sites)

    assert isinstance(top_orbitals, list)
    assert all(isinstance(idx, int) for idx in top_orbitals)
    assert all(0 <= idx < n_sites for idx in top_orbitals)


@pytest.mark.parametrize("molecule_file", MOLECULE_FILES)
def test_ef_active_space_brute_force(molecule_file):
    mol_data, mps = build_mps_from_molecule(molecule_file)
    n_orbs = mol_data.num_spin_orbs // 2
    n_sites = min(nactive, n_orbs)

    active_orbitals = ef_active_space_brute_force(mps, n_sites=n_sites)

    assert isinstance(active_orbitals, list)
    assert all(isinstance(idx, int) for idx in active_orbitals)
    print(f"Selected active orbitals for {molecule_file}: {active_orbitals} using EF brute-force method")

@pytest.mark.parametrize("molecule_file", MOLECULE_FILES)
def test_ef_active_space_greedy(molecule_file):
    mol_data, mps = build_mps_from_molecule(molecule_file)
    n_orbs = mol_data.num_spin_orbs // 2
    n_sites = min(nactive, n_orbs)

    active_orbitals = ef_active_space_greedy(mps, n_sites=n_sites)

    assert isinstance(active_orbitals, list)
    assert all(isinstance(idx, int) for idx in active_orbitals)
    print(f"Selected active orbitals for {molecule_file}: {active_orbitals} using EF greedy method")

@pytest.mark.parametrize("molecule_file", MOLECULE_FILES)
def test_ef_active_space_sample(molecule_file):
    mol_data, mps = build_mps_from_molecule(molecule_file)
    n_orbs = mol_data.num_spin_orbs // 2
    n_sites = min(nactive, n_orbs)

    active_orbitals = ef_active_space_sample(mps, n_sites=n_sites)

    assert isinstance(active_orbitals, list), f"Expected list of active orbitals, got {type(active_orbitals)}"
    assert all(isinstance(idx, int) for idx in active_orbitals), f"Expected all indices to be integers, got {[type(idx) for idx in active_orbitals]}"
    print(f"Selected active orbitals for {molecule_file}: {active_orbitals} using EF sample method")
#-------------------------------------------------------------------------------

# -------------------------------------------------------------------------------


# Run HF:
def run_hf(molecule_file: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    location = os.path.join(repo_root, "molecules/mol_info", molecule_file)
    with open(location) as f:
        info = json.load(f)
        mol_string = info["xyz"]
        mol_basis = "STO-3G"
        mol_charge = info["charge"]

    mol_obj = pyscf.M(
        atom=mol_string,
        basis=mol_basis,
        charge=mol_charge,
    )
    rhf_obj = pyscf.scf.RHF(mol_obj).run()
    return rhf_obj


def run_casci_auto(rhf_obj, nactive):
    n_electrons = int(rhf_obj.mo_occ.sum())
    n_occ = n_electrons // 2  # number of doubly occupied orbitals (RHF)

    # active space spans nactive orbitals centred on HOMO-LUMO
    # number of occupied orbitals inside active space
    n_occ_in_active = min(nactive, n_occ)  # can't have more than total occupied
    nelec = 2 * n_occ_in_active

    casci_obj = pyscf.mcscf.CASCI(rhf_obj, nactive, nelec)
    casci_result = casci_obj.kernel()
    return casci_result[0]


# Run CASCI using a built-in pyscf active space selection method (AVAS - not TN based)
def run_casci_avas(rhf_obj):
    ncas, nelecas, mo = avas.avas(
        rhf_obj,
        ["N 2p"],  # target atomic orbitals for AVAS
    )
    mc = pyscf.mcscf.CASCI(rhf_obj, ncas, nelecas)
    casci_result = mc.kernel(mo)
    energy = casci_result[0]
    return energy


# Run CASCI with your selected orbitals
def run_casci_selected(rhf_obj, active_space):
    norb = len(active_space)
    # mo_occ = rhf_obj.mo_occ
    nelec = int(rhf_obj.mo_occ[np.array(active_space) - 1].sum())

    # if not all(0 <= i < norb for i in active_space):
    # raise IndexError(f"Active-space indices out of range: {active_space}")

    if len(set(active_space)) != len(active_space):
        raise ValueError("Active-space indices must be unique")

    casci_obj = pyscf.mcscf.CASCI(rhf_obj, norb, nelec)
    orbs_casci = casci_obj.sort_mo(active_space)
    casci_result = casci_obj.kernel(orbs_casci)

    return casci_result[0]

def run_autocas(molecule_file, n_active):
    mol_data, mps = build_mps_from_molecule(molecule_file)

    n_sites = mol_data.num_spin_orbs // 2
    active_orbitals = min(n_active, n_sites)

    top_orbitals = autocas_selection_ranked_fixed_number(
        mps, n_sites=n_sites, active_orbitals=active_orbitals
    )
    return top_orbitals


def run_ef_brute_force(molecule_file, n_active):
    mol_data, mps = build_mps_from_molecule(molecule_file)
    n_orbs = mol_data.num_spin_orbs // 2
    n_sites = min(n_active, n_orbs)

    active_orbitals = ef_active_space_brute_force(mps, n_sites=n_sites)
    return active_orbitals

def run_ef_greedy(molecule_file, n_active):
    mol_data, mps = build_mps_from_molecule(molecule_file)
    n_orbs = mol_data.num_spin_orbs // 2
    n_sites = min(n_active, n_orbs)

    active_orbitals = ef_active_space_greedy(mps, n_sites=n_sites)
    return active_orbitals

def run_ef_sample(molecule_file, n_active):
    mol_data, mps = build_mps_from_molecule(molecule_file)
    n_orbs = mol_data.num_spin_orbs // 2
    n_sites = min(n_active, n_orbs)

    active_orbitals = ef_active_space_sample(mps, n_sites=n_sites)
    return active_orbitals

def test_benchmark():
    rhf_obj = run_hf("N2.json")
    print(rhf_obj.mo_coeff.shape)
    nactive = 6
    energy_auto = run_casci_auto(rhf_obj, nactive)
    energy_avas = run_casci_avas(rhf_obj)
    autocas_selected = run_autocas("N2.json", nactive)
    autocas_selected = [x + 1 for x in autocas_selected]
    ef_selected = run_ef_brute_force("N2.json", nactive)
    ef_selected = [x + 1 for x in ef_selected]
    greedy_selected = run_ef_greedy("N2.json", nactive)
    greedy_selected = [x + 1 for x in greedy_selected]
    sample_selected = run_ef_sample("N2.json", nactive)
    sample_selected = [x + 1 for x in sample_selected]
    print(f"AutoCAS-selected active space: {autocas_selected}")
    print(f"Brute-force EF-selected active space: {ef_selected}")
    print(f"Greedy EF-selected active space: {greedy_selected}")
    print(f"Sample EF-selected active space: {sample_selected}")
    energy_selected = run_casci_selected(rhf_obj, active_space=ef_selected)
    energy_autocas = run_casci_selected(rhf_obj, active_space=autocas_selected)
    energy_greedy = run_casci_selected(rhf_obj, active_space=greedy_selected)
    energy_sample = run_casci_selected(rhf_obj, active_space=sample_selected)
    print(f"Energy with auto-selected active space: {energy_auto}")
    print(f"Energy with AVAS-selected active space: {energy_avas}")
    print(f"Energy with brute-force selected active space: {energy_selected}")
    print(f"Energy with AutoCAS-selected active space: {energy_autocas}")
    print(f"Energy with greedy EF-selected active space: {energy_greedy}")
    print(f"Energy with sample EF-selected active space: {energy_sample}")