import numpy as np
from tn4qa.mps import MatrixProductState as MPS
from tn4qa.tn_methods.entanglement_feature import build_entanglement_feature, contract_ef_bitstring, ef_best_cut, split_mps_at_cut

TEST_ARRAYS = [
    np.random.rand(4, 2, 1),
    np.random.rand(4, 6, 2),
    np.random.rand(6, 2, 1),
]

def test_build_entanglement_feature_shapes():
    mps = MPS.from_arrays(TEST_ARRAYS)
    EF = build_entanglement_feature(mps)
    assert len(EF) == len(mps.tensors)

    for site, T in enumerate(EF):
        # T shape = (2, u, u, p, d, d, p)
        print(T.data.shape)
        assert T.ndim == 7
        assert T.data.shape[0] == 2
        print(T.data)
        u = [4, 4, 6]
        p = [1, 2, 1]
        d = [2, 6, 2]
        for i in range(len(u)):
            if site == i:
                assert T.data.shape[1] == u[i]
                assert T.data.shape[2] == u[i]
                assert T.data.shape[3] == p[i]
                assert T.data.shape[4] == d[i]
                assert T.data.shape[5] == d[i]
                assert T.data.shape[6] == p[i]
    print("build_entanglement_feature_shapes: PASSED")


def test_contract_ef_bitstring_identity_only():
    EF = [
        [np.ones((1, 1, 1, 1, 1, 1))],  # site 0, bit 0
        [np.ones((1, 1, 1, 1, 1, 1))],  # site 1, bit 0
    ]
    bitstring = [0, 0]
    R2 = contract_ef_bitstring(EF, bitstring)
    assert np.isscalar(R2)
    assert np.isfinite(R2)
    assert R2 > 0
    expected = 1.0
    assert np.isclose(R2, expected)
    print("contract_ef_bitstring_identity_only: PASSED")

def test_contract_ef_bitstring_identity_small():
    """
    Test contract_ef_bitstring on a 2-site MPS with small bond and physical dimensions.
    EF tensors are all ones, so the expected R2 is the product of the diagonal dimensions
    after reshaping each EF tensor to 2D.
    """
    # 2-site EF MPO
    EF = [
        [np.ones((2, 2, 2, 2, 2, 2))],  # site 0, bit 0: shape (uL,uR,p,dL,dR,p)
        [np.ones((3, 3, 1, 3, 3, 1))],  # site 1, bit 0
    ]
    bitstring = [0, 0]

    # Compute R2 using contract_ef_bitstring
    R2 = contract_ef_bitstring(EF, bitstring)

    # Expected value: product of traces of each site (since all ones)
    # trace for a site = sum of diagonal elements = min(left_dim, right_dim) * value_per_element
    # Here all elements = 1, left_dim = prod(shape[:3]), right_dim = prod(shape[3:])
    expected = 1
    for i, T in enumerate(EF):
        Ti = T[0]
        left = np.prod(Ti.shape[:3])
        right = np.prod(Ti.shape[3:])
        # Since all ones, trace = min(left, right) * 1
        expected *= min(left, right)

    print("R2:", R2, "Expected:", expected)

    # Assertions
    assert np.isscalar(R2)
    assert np.isfinite(R2)
    assert R2 > 0
    assert np.isclose(R2, expected)
    print("test_contract_ef_bitstring_identity_small: PASSED")

import numpy as np

def test_contract_ef_bitstring_nonidentity():
    """
    Test contract_ef_bitstring with a small 2-site EF MPO that is NOT identity.
    We use simple integers so the expected R2 can be computed manually.
    """
    # EF tensors (2 sites, 1 bit per site)
    # Shapes: (uL, uR, p, dL, dR, p)
    EF = [
        [np.array([[[[[[1]]]]]])],  # site 0, bit 0: 1x1x1x1x1x1
        [np.array([[[[[[2]]]]]])],  # site 1, bit 0: 1x1x1x1x1x1
    ]
    bitstring = [0, 0]

    # Compute R2
    R2 = contract_ef_bitstring(EF, bitstring)

    # Expected R2 = trace(T0) * trace(T1) = 1 * 2 = 2
    expected = 2.0

    print("R2:", R2, "Expected:", expected)

    # Assertions
    assert np.isscalar(R2)
    assert np.isfinite(R2)
    assert R2 > 0
    assert np.isclose(R2, expected)
    print("test_contract_ef_bitstring_nonidentity: PASSED")

def test_contract_ef_bitstring_mixed():
    mps = MPS.from_arrays(TEST_ARRAYS)
    EF = build_entanglement_feature(mps)
    bitstring = [1, 0, 0]
    R2 = contract_ef_bitstring(EF, bitstring)
    print("R2:", R2)
    assert np.isscalar(R2)
    assert np.isfinite(R2)
    assert np.isreal(R2)
    assert R2 > 0
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
    num_qubits = 20           
    mps = MPS.random_quantum_state_mps(num_qubits, 2, 2)
    cut = ef_best_cut(mps)
    assert 1 <= cut < num_qubits
    print("best_cut_on_random_mps: PASSED")

    
if __name__ == "__main__":
    test_build_entanglement_feature_shapes()
    test_contract_ef_bitstring_identity_only()
    test_contract_ef_bitstring_identity_small()
    test_contract_ef_bitstring_nonidentity()
    test_contract_ef_bitstring_mixed()
    test_ef_best_cut_bounds()
    test_split_mps_at_cut()
    test_best_cut_on_random_mps()