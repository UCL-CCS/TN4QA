import os

import numpy as np
from pyscf import gto, scf

from tn4qa.dmrg import DMRG, Block2FermionDMRG
from tn4qa.utils import ReadMoleculeData

np.random.seed(1)
cwd = os.getcwd()


def test_Block2FermionDMRG_RHF():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    dmrg = Block2FermionDMRG(mf, "RHF", 256)
    energy = dmrg.run(20)
    assert np.isclose(energy, -107.654122447524472)


def test_Block2FermionDMRG_UHF():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    dmrg = Block2FermionDMRG(mf, "UHF", 256)
    energy = dmrg.run(20)
    assert np.isclose(energy, -107.654122447524472)


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


def test_DMRG_subspace_expansion():
    location = os.path.join(cwd, "molecules/H2.json")
    mol_data = ReadMoleculeData(location)
    ham = mol_data.qubit_hamiltonian
    dmrg = DMRG(ham, 16, "subspace-expansion")
    energy, _ = dmrg.run(10)
    assert np.isclose(energy, -2.8625885726691855, atol=0.1)
