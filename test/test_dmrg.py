import os

import numpy as np

from tn4qa.dmrg import DMRG
from tn4qa.utils import ReadMoleculeData

np.random.seed(1)
cwd = os.getcwd()


def test_DMRG_one_site():
    location = os.path.join(cwd, "molecules/N2.json")
    mol_data = ReadMoleculeData(location)
    ham = mol_data.qubit_hamiltonian
    dmrg = DMRG(ham, 4, method="one-site")
    energy, _ = dmrg.run(4)
    assert np.isclose(energy, -107.65412244752251, atol=1.0)


def test_DMRG_two_site():
    location = os.path.join(cwd, "molecules/LiH.json")
    mol_data = ReadMoleculeData(location)
    ham = mol_data.qubit_hamiltonian
    dmrg = DMRG(ham, 8, method="two-site")
    energy, _ = dmrg.run(10)
    assert np.isclose(energy, -7.881571973351853, atol=0.1)


def test_DMRG_fermionic():
    location = os.path.join(cwd, "molecules/H2.json")
    mol_data = ReadMoleculeData(location)
    ham = mol_data.fermionic_hamiltonian
    dmrg = DMRG(ham, 8, method="two-site")
    energy, _ = dmrg.run(10)
    assert np.isclose(energy, -1.10115033023, atol=0.01)
