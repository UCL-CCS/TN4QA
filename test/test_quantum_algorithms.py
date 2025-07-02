import os

import numpy as np

from tn4qa.quantum_algorithms.variational.vqe import VQEAlgorithm
from tn4qa.utils import ReadMoleculeData


def test_vqe_h2():
    cwd = os.getcwd()
    location = os.path.join(cwd, "molecules/H2.json")
    mol_data = ReadMoleculeData(location)
    ham = mol_data.qubit_hamiltonian
    hf_e = mol_data.rhf_energy
    vqe = VQEAlgorithm(ham, 2)
    result = vqe.run()
    assert np.isclose(result.result, hf_e)
