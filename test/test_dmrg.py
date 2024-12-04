import numpy as np
from pyscf import gto, scf
from tn4qa.dmrg import FermionDMRG, QubitDMRG

np.random.seed(1)

def test_FermionDMRG_RHF():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1E-14)
    print("hi")
    dmrg = FermionDMRG(mf, "RHF", 512, 256)
    print("bye")
    energy = dmrg.run(20)
    assert np.isclose(energy, -107.654122447524472)

def test_FermionDMRG_UHF():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1E-14)
    dmrg = FermionDMRG(mf, "UHF", 512, 256)
    energy = dmrg.run(20)
    assert np.isclose(energy, -107.654122447524472)
