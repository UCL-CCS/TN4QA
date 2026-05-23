import os

import numpy as np

from tn4qa.dmrg import DMRG
from tn4qa.mpo import MatrixProductOperator
from tn4qa.mps import MatrixProductState
from tn4qa.utils import ReadMoleculeData

np.random.seed(10)
cwd = os.getcwd()


def test_DMRG_nitrogen():
    location = os.path.join(cwd, "molecules/N2.json")
    mol_data = ReadMoleculeData(location)
    ham = mol_data.qubit_hamiltonian
    hf_mps = MatrixProductState.from_hf_state(
        mol_data.num_spin_orbs, mol_data.num_electrons
    )
    dmrg = DMRG(ham, 32, hf_mps)
    energy, _ = dmrg.run(30)
    assert np.isclose(energy, -107.54930095790698, atol=0.01)


def test_DMRG_LiH():
    location = os.path.join(cwd, "molecules/LiH.json")
    mol_data = ReadMoleculeData(location)
    ham = mol_data.qubit_hamiltonian
    dmrg = DMRG(ham, 16, None)
    energy, _ = dmrg.run(20)
    assert np.isclose(energy, -7.784460280031227, atol=0.01)


def test_DMRG_hydrogen():
    location = os.path.join(cwd, "molecules/H2.json")
    mol_data = ReadMoleculeData(location)
    ham = mol_data.qubit_hamiltonian
    hf_mps = MatrixProductState.from_hf_state(
        mol_data.num_spin_orbs, mol_data.num_electrons
    )
    dmrg = DMRG(ham, 16, hf_mps)
    energy, psi = dmrg.run(20)

    ham_mpo = MatrixProductOperator.from_hamiltonian(ham)
    ip = psi.compute_expectation_value(ham_mpo)

    assert np.isclose(ip.real, -1.10115033023, atol=0.01)
    assert np.isclose(energy, -1.10115033023, atol=0.01)
