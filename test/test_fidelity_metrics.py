import numpy as np

from tn4qa.fidelity_metrics import state_uhlmann_fidelity, total_variation_distance
from tn4qa.mps import MatrixProductState


def test_uhlmann_fid_pure_states():
    psi = MatrixProductState.equal_superposition_mps(5)
    phi = MatrixProductState.from_bitstring("11111")
    uhlmann_fid = state_uhlmann_fidelity(psi, phi)

    # Expect 1/32
    assert np.isclose(uhlmann_fid, 1.0 / 32)


def test_uhlmann_fid_states():
    psi = MatrixProductState.from_bitstring("0000")
    phi = MatrixProductState.from_bitstring("1111")
    psi_dm = psi.form_density_operator()
    phi_dm = phi.form_density_operator()
    uhlmann_fid = state_uhlmann_fidelity(psi_dm, phi_dm)

    # Expect 0
    assert np.isclose(uhlmann_fid, 0.0)


def test_total_variation_distance_exact():
    expected_dist = {bin(k)[2:].zfill(3): (1 / 2**3) for k in range(8)}
    psi = MatrixProductState.equal_superposition_mps(3)
    tvd = total_variation_distance(psi, expected_dist)

    assert np.isclose(tvd, 0.0)


def test_total_variation_distance_approx():
    expected_dist = {bin(k)[2:].zfill(5): (1 / 2**5) for k in range(32)}
    psi = MatrixProductState.equal_superposition_mps(5)
    tvd = total_variation_distance(psi, expected_dist, sample_size=1000)

    assert np.isclose(tvd, 0.0, atol=0.1)
