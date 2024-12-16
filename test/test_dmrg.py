import json
import os

import numpy as np
from pyscf import gto, scf

from tn4qa.dmrg import FermionDMRG, QubitDMRG

np.random.seed(2)
cwd = os.getcwd()


def test_FermionDMRG_RHF():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    dmrg = FermionDMRG(mf, "RHF", 256)
    energy = dmrg.run(20)
    assert np.isclose(energy, -107.654122447524472)


def test_FermionDMRG_UHF():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    dmrg = FermionDMRG(mf, "UHF", 256)
    energy = dmrg.run(20)
    assert np.isclose(energy, -107.654122447524472)


def test_QubitDMRG():
    location = os.path.join(cwd, "hamiltonians/HeH.json")
    with open(location) as f:
        ham = json.load(f)
    ham_dict = {k: float(v[0]) for k, v in ham.items()}
    dmrg = QubitDMRG(ham_dict, 4)
    energy, _ = dmrg.run(2)
    assert np.isclose(energy, -2.8625885726691855, atol=1.0)
