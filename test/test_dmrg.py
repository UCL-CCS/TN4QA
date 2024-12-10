import numpy as np
from pyscf import gto, scf
from tn4qa.dmrg import FermionDMRG, QubitDMRG
import json
from timeit import default_timer
import os

np.random.seed(2)
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
    location = os.path.join(cwd, "hamiltonians/HeH.json")
    with open(location, "r") as f:
        ham = json.load(f)
    ham_dict = {k : float(v[0]) for k,v in ham.items()}
    dmrg = QubitDMRG(ham_dict, np.infty, 4) # Currently no MPO truncation is implemented so np.infty is fine
    energy, _ = dmrg.run(2)
    assert np.isclose(energy, -2.8625885726691855, atol=1.0)

def test_timings():
    times = []
    energies = []

    start = default_timer()
    location = os.path.join(cwd, "hamiltonians/HeH.json")
    with open(location, "r") as f:
        ham = json.load(f)
    ham_dict = {k : float(v[0]) for k,v in ham.items()}
    dmrg = QubitDMRG(ham_dict, np.infty, 4) # Currently no MPO truncation is implemented so np.infty is fine
    energy, _ = dmrg.run(2)
    energies.append(energy)
    stop1 = default_timer()
    times.append(stop1-start)

    location = os.path.join(cwd, "hamiltonians/LiH.json")
    with open(location, "r") as f:
        ham = json.load(f)
    ham_dict = {k : float(v[0]) for k,v in ham.items()}
    dmrg = QubitDMRG(ham_dict, np.infty, 4) # Currently no MPO truncation is implemented so np.infty is fine
    energy, _ = dmrg.run(2)
    energies.append(energy)
    stop2 = default_timer()
    times.append(stop2-stop1)

    location = os.path.join(cwd, "hamiltonians/N2.json")
    with open(location, "r") as f:
        ham = json.load(f)
    ham_dict = {k : float(v) for k,v in ham.items()}
    dmrg = QubitDMRG(ham_dict, np.infty, 4) # Currently no MPO truncation is implemented so np.infty is fine
    energy, _ = dmrg.run(2)
    energies.append(energy)
    stop3 = default_timer()
    times.append(stop3-stop1)

    dict = {"HeH" : {"Time" : times[0], "Energy" : energies[0]}, "LiH" : {"Time" : times[1], "Energy" : energies[1]}, "N2" : {"Time" : times[2], "Energy" : energies[2]}}

    with open("one_site_method.json", "w") as f:
        json.dump(dict, f)
