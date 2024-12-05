import numpy as np
from pyscf import gto, scf
from tn4qa.dmrg import FermionDMRG, QubitDMRG
import json
import os

np.random.seed(1)
cwd = os.getcwd()

def test_FermionDMRG_RHF():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1E-14)
    dmrg = FermionDMRG(mf, "RHF", 512, 256)
    energy = dmrg.run(20)
    assert np.isclose(energy, -107.654122447524472)

def test_FermionDMRG_UHF():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1E-14)
    dmrg = FermionDMRG(mf, "UHF", 512, 256)
    energy = dmrg.run(20)
    assert np.isclose(energy, -107.654122447524472)

def test_QubitDMRG():
    location = os.path.join(cwd, "hamiltonians/N2.json")
    with open(location, "r") as f:
        ham = json.load(f)
    dmrg = QubitDMRG(ham, np.infty, 4) # Currently no MPO truncation is implemented so np.infty is fine
    energy, _ = dmrg.run(2)
    assert np.isclose(energy, -107.654122447524472, atol=1.0)
