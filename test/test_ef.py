import numpy as np
from tn4qa.mps import MatrixProductState as MPS
from tn4qa.tn_methods.entanglement_feature import build_entanglement_feature, contract_ef_bitstring, ef_best_cut, split_mps_at_cut

TEST_ARRAYS = [
    np.random.rand(4, 2),
    np.random.rand(4, 6, 2),
    np.random.rand(6, 2),
]

def test_build_entanglement_feature_shapes():
    mps = MPS.from_arrays(TEST_ARRAYS)
    EF = build_entanglement_feature(mps)
    assert len(EF) == len(mps.tensors)

    for site, T in enumerate(EF):
        # T shape = (2, u, u, p, d, d, p)
        assert T.ndim == 7
        assert T.data.shape[0] == 2
        u, d, p = mps.tensors[site].data.shape[:3]
        assert T.data.shape[1] == u
        assert T.data.shape[2] == u
        assert T.data.shape[3] == p
        assert T.data.shape[4] == d
        assert T.data.shape[5] == d
        assert T.data.shape[6] == p
    print("build_entanglement_feature_shapes: PASSED")


def test_contract_ef_bitstring_identity_only():
    mps = MPS.from_arrays(TEST_ARRAYS)
    EF = build_entanglement_feature(mps)
    bitstring = [0] * len(EF)
    R2 = contract_ef_bitstring(mps, EF, bitstring)

    # Expected Renyi-2 is product of EF dimensions along the diagonal
    expected = np.prod([T.data.shape[1] for T in EF])
    assert np.isclose(R2, expected)
    print("contract_ef_bitstring_identity_only: PASSED")

def test_contract_ef_bitstring_mixed():
    mps = MPS.from_arrays(TEST_ARRAYS)
    EF = build_entanglement_feature(mps)
    bitstring = [1, 0, 0]
    R2 = contract_ef_bitstring(mps, EF, bitstring)
    assert np.isscalar(R2)
    assert np.isfinite(R2)
    print("contract_ef_bitstring_mixed: PASSED")

def test_ef_best_cut_bounds():
    mps = MPS.from_arrays(TEST_ARRAYS)
    cut = ef_best_cut(mps)
    assert 1 <= cut < len(TEST_ARRAYS)
    print("ef_best_cut_bounds: PASSED")

def test_split_mps_at_cut():
    mps = MPS.from_arrays(TEST_ARRAYS)
    cut = 1
    left, right = split_mps_at_cut(mps, cut)
    assert len(left.tensors) == 1
    assert len(right.tensors) == 2
    print("split_mps_at_cut: PASSED")


def test_best_cut_on_random_mps():
    num_qubits = 4            # 6 seems to be the cut off point, after which it explodes
    mps = MPS.random_quantum_state_mps(num_qubits, 2, 2)
    cut = ef_best_cut(mps)
    assert 1 <= cut < num_qubits
    print("best_cut_on_random_mps: PASSED")

    
if __name__ == "__main__":
    test_build_entanglement_feature_shapes()
    test_contract_ef_bitstring_identity_only()
    test_contract_ef_bitstring_mixed()
    test_ef_best_cut_bounds()
    test_split_mps_at_cut()
    test_best_cut_on_random_mps()