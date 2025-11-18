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
        dim = T.shape[1]  # EF shape is (2, dim, dim)
        assert T.shape == (2, dim, dim)
    print("build_entanglement_feature_shapes: PASSED")


def test_contract_ef_bitstring_identity_only():
    mps = MPS.from_arrays(TEST_ARRAYS)
    EF = build_entanglement_feature(mps)
    bitstring = [0] * len(EF)
    R2 = contract_ef_bitstring(EF, bitstring)

    # Expected Renyi-2 is product of EF dimensions along the diagonal
    expected = np.prod([T.shape[1] for T in EF])
    assert np.isclose(R2, expected)
    print("contract_ef_bitstring_identity_only: PASSED")

def test_contract_ef_bitstring_mixed():
    mps = MPS.from_arrays(TEST_ARRAYS)
    EF = build_entanglement_feature(mps)
    bitstring = [1, 0, 0]
    R2 = contract_ef_bitstring(EF, bitstring)
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

if __name__ == "__main__":
    test_build_entanglement_feature_shapes()
    test_contract_ef_bitstring_identity_only()
    test_contract_ef_bitstring_mixed()
    test_ef_best_cut_bounds()
    test_split_mps_at_cut()