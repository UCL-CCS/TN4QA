import json

import numpy as np

from tn4qa.dmrg import DMRG
from tn4qa.qi_metrics import get_one_orbital_rdm, get_two_orbital_rdm

with open("test/data/h2_rdm.json") as f:
    data = json.load(f)
    h2_rdm1 = data["RDM1"]
    h2_rdm2 = data["RDM2"]

with open("molecules/H2.json") as f:
    h2_data = json.load(f)
    h2_ham = h2_data["qubit_hamiltonian"]
    h2_ham = {k: v[0] + 1j * v[1] for k, v in h2_ham.items()}

h2_dmrg = DMRG(h2_ham, max_mps_bond=4, method="two-site")
h2_dmrg.run(20)
h2_mps = h2_dmrg.mps


def test_rdm1():
    h2_rdm1_dmrg = get_one_orbital_rdm(h2_mps, 1)

    assert np.allclose(h2_rdm1, h2_rdm1_dmrg, atol=0.01)


def test_rdm2():
    h2_rdm2_dmrg = get_two_orbital_rdm(h2_mps, [1, 2])

    assert np.allclose(h2_rdm2, h2_rdm2_dmrg, atol=0.01)
