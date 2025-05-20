import json

import numpy as np

from tn4qa.dmrg import DMRG
from tn4qa.qi_metrics import get_one_orbital_rdm, get_two_orbital_rdm
from tn4qa.utils import ReadMoleculeData

with open("test/data/h2_rdm.json") as f:
    data = json.load(f)
    h2_rdm1 = data["RDM1"]
    h2_rdm1 = [[n[0] + 1j * n[1] for n in row] for row in h2_rdm1]
    h2_rdm2 = data["RDM2"]
    h2_rdm2 = [[n[0] + 1j * n[1] for n in row] for row in h2_rdm2]

h2_file = "molecules/H2.json"
h2_data = ReadMoleculeData(h2_file)
h2_ham = h2_data.qubit_hamiltonian

h2_dmrg = DMRG(h2_ham, max_mps_bond=4, method="two-site")
h2_dmrg.run(20)
h2_mps = h2_dmrg.mps


def test_rdm1():
    h2_rdm1_dmrg = get_one_orbital_rdm(h2_mps, 1)

    assert np.allclose(h2_rdm1, h2_rdm1_dmrg, atol=0.01)


def test_rdm2():
    h2_rdm2_dmrg = get_two_orbital_rdm(h2_mps, [1, 2])

    assert np.allclose(h2_rdm2, h2_rdm2_dmrg, atol=0.01)
