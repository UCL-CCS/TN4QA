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
    dmrg = DMRG(ham, 4, method="two-site")
    energy, _ = dmrg.run(10)
    assert np.isclose(energy, -7.881571973351853, atol=0.1)
