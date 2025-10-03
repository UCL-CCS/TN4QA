import os
import random

import numpy as np
from qiskit import QuantumCircuit

from tn4qa.quantum_algorithms.phase_estimation.hadamard_test import HadamardTest
from tn4qa.quantum_algorithms.phase_estimation.qpe import QPE
from tn4qa.quantum_algorithms.variational.vqe import VQEAlgorithm
from tn4qa.utils import ReadMoleculeData

seed = random.seed(1)


def test_vqe_h2():
    cwd = os.getcwd()
    location = os.path.join(cwd, "molecules/H2.json")
    mol_data = ReadMoleculeData(location)
    ham = mol_data.qubit_hamiltonian
    hf_e = mol_data.rhf_energy
    vqe = VQEAlgorithm(ham, 2)
    result = vqe.run()
    assert np.isclose(result.result, hf_e, atol=0.001)


def test_hadamard_test():
    state = QuantumCircuit(3)
    state.h([0, 1, 2])
    unitary = QuantumCircuit(3)
    unitary.x([0, 1, 2])

    htest = HadamardTest(unitary, state)
    result = htest.run(10000)
    assert np.isclose(result.result.real, 1.0)
    assert result.result.imag < 0.02


def test_qpe_1q():
    unitary = QuantumCircuit(1)
    unitary.z(0)

    state = QuantumCircuit(1)
    state.x(0)

    qpe = QPE(unitary, state, 2)
    result = qpe.run(1024)
    assert result.result == 0.5
