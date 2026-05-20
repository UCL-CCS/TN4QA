import numpy as np
import pytest
import os

from tn4qa.mps import MatrixProductState as MPS
from tn4qa.qi_metrics import get_one_orbital_entropy
from test.test_active_space import build_mps_from_molecule
from tn4qa.tn_methods.entanglement_feature import (
    build_entanglement_feature,
    contract_ef_bitstring,
    ef_best_cut,
    split_mps_at_cut,
)

TEST_ARRAYS = [
    np.random.rand(4, 2),
    np.random.rand(4, 2, 2),
    np.random.rand(2, 2),
]

mps = MPS.from_arrays(TEST_ARRAYS)
ef_mps = build_entanglement_feature(mps)


def test_build_entanglement_feature_shapes():
    for A in ef_mps.tensors:
        if A.data.ndim == 2:
            assert A.data.shape == (256, 2)
        elif A.data.ndim == 3:
            assert A.data.shape == (256, 256, 2)
    print("build_entanglement_feature_shapes: PASSED")


def test_contract_ef_bitstring():
    R2 = contract_ef_bitstring(ef_mps, [0] * len(TEST_ARRAYS))
    assert R2 > 0
    assert isinstance(R2, float)
    N = len(ef_mps.tensors)
    for i in range(1, N):
        bitstring = [1] * i + [0] * (N - i)
        bit_mps = MPS.from_bitstring(bitstring)

        env = None
        for A, B in zip(ef_mps.tensors, bit_mps.tensors):
            A = A.data if hasattr(A, "data") else A
            B = B.data if hasattr(B, "data") else B

            C = np.tensordot(A.conj(), B, axes=([-1], [-1]))
            C = np.squeeze(C)
            if env is None:
                env = C
            else:
                env = np.tensordot(env, C, axes=([-1], [0]))

        if hasattr(env, "todense"):
            val = env.todense()
        else:
            val = np.asarray(env)
        R2_test = float(val.real)
        assert np.isclose(R2, R2_test)
    print("contract_ef_bitstring_identity_only: PASSED")


def test_ef_best_cut_bounds():
    cut = ef_best_cut(ef_mps)
    assert 1 <= cut < len(TEST_ARRAYS)
    print("ef_best_cut_bounds: PASSED")


def test_split_mps_at_cut():
    cut = 1
    left, right = split_mps_at_cut(mps, cut)
    assert len(left.tensors) == 1
    assert len(right.tensors) == 2
    print("split_mps_at_cut: PASSED")


def test_best_cut_on_random_mps():
    num_qubits = 20
    test_mps = MPS.random_quantum_state_mps(num_qubits, 2, 2)
    cut = ef_best_cut(test_mps)
    assert 1 <= cut < num_qubits
    print("best_cut_on_random_mps: PASSED")

MOLECULE_FILES = ["H2.json", "LiH.json", "N2.json"]

@pytest.mark.parametrize("molecule_file", MOLECULE_FILES)
def test_bitstrings_with_one_one(molecule_file):
    mol_data, mps = build_mps_from_molecule(molecule_file)

    n_sites = mps.num_sites
    n_orbitals = mol_data.num_spin_orbs // 2
    print(f"Testing molecule {molecule_file} with {n_sites} sites and {n_orbitals} orbitals") 

    # Calculate one-orbital entropy for each orbital
    entropies = []
    for i in range(1, n_orbitals + 1):
        entropy = get_one_orbital_entropy(mps, i)
        entropies.append(entropy)

    # Build EF from DMRG solution
    ef_mps = build_entanglement_feature(mps)

    # contract EF with bitstrings with exactly one '1'
    for i in range(n_orbitals):
        bitstring = [0] * n_sites
        bitstring[2 * i] = 1
        bitstring[2 * i + 1] = 1
        R2 = contract_ef_bitstring(ef_mps, bitstring)
        assert R2 > 0
        print(f"Orbital {i+1} entropy: {entropies[i]:.4f}, R2: {R2:.4f}")

# if __name__ == "__main__":
#     test_build_entanglement_feature_shapes()
#     test_contract_ef_bitstring()
#     test_ef_best_cut_bounds()
#     test_split_mps_at_cut()
#     test_best_cut_on_random_mps()
