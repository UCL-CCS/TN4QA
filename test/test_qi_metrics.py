import json

import numpy as np

from tn4qa.dmrg import DMRG
from tn4qa.mps import MatrixProductState
from tn4qa.qi_metrics import (
    get_all_mutual_information,
    get_mutual_information,
    get_one_orbital_entropy,
    get_one_orbital_rdm,
    get_two_orbital_rdm,
)
from tn4qa.utils import ReadMoleculeData

with open("test/data/h2_rdm.json") as f:
    data = json.load(f)
    h2_rdm1 = data["RDM1"]
    h2_rdm1 = np.array([[n[0] + 1j * n[1] for n in row] for row in h2_rdm1])
    h2_rdm2 = data["RDM2"]
    h2_rdm2 = np.array([[n[0] + 1j * n[1] for n in row] for row in h2_rdm2])

h2_file = "molecules/H2.json"
h2_data = ReadMoleculeData(h2_file)
h2_ham = h2_data.qubit_hamiltonian

hf_state = MatrixProductState.from_hf_state(4, 2)
h2_dmrg = DMRG(h2_ham, 16, hf_state)
_, h2_mps = h2_dmrg.run(20)

lih_file = "molecules/LiH.json"
lih_data = ReadMoleculeData(lih_file)
lih_ham = lih_data.qubit_hamiltonian

hf_state = MatrixProductState.from_hf_state(12, 4)
lih_dmrg = DMRG(lih_ham, 16)
_, lih_mps = lih_dmrg.run(20)


def test_rdm1():
    h2_rdm1_dmrg = get_one_orbital_rdm(h2_mps, 1)
    print(h2_rdm1)
    print(h2_rdm1_dmrg)

    assert np.allclose(h2_rdm1, h2_rdm1_dmrg, atol=0.01)


def test_rdm2():
    h2_rdm2_dmrg = get_two_orbital_rdm(h2_mps, [1, 2])
    print(h2_rdm2.round(2))
    print(h2_rdm2_dmrg.round(2))

    assert np.allclose(h2_rdm2, h2_rdm2_dmrg, atol=0.01)


def test_mutual_information():
    assert np.isclose(
        get_mutual_information(lih_mps, [1, 1]),
        get_one_orbital_entropy(lih_mps, 1),
        atol=1e-4,
    )


def test_get_all_mutual_information():
    water_mi = get_all_mutual_information(lih_mps)
    assert np.allclose(
        [
            get_one_orbital_entropy(lih_mps, i + 1)
            for i in range(np.diag(water_mi).size)
        ],
        np.diag(water_mi),
        atol=1e-4,
    )
    # allow for a small negative which approximates zero
    assert np.all(water_mi >= -1e-8)
    assert np.all(water_mi == water_mi.T)
