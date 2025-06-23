"""Test configuraion."""

import pickle
from pathlib import Path

import pytest

from tn4qa.dmrg import DMRG


@pytest.fixture(scope="module")
def water_integrals():
    folder = Path(__file__).parent
    with open(folder.joinpath("./data/water_ones.pkl"), "rb") as file:
        ones = pickle.load(file)

    with open(folder.joinpath("./data/water_twos.pkl"), "rb") as file:
        twos = pickle.load(file)
        # water_twos file here includes factor of 0.5 but this is accounted for in the MPO builder
        twos = 2 * twos
    return (ones, twos)


@pytest.fixture(scope="module")
def water_DMRG(water_integrals):
    ones, twos = water_integrals
    dmrg = DMRG(hamiltonian=(ones, twos, 0), max_mps_bond=4, method="two-site")
    dmrg.run(5)
    return dmrg.mps
