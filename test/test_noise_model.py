"""Test for the noise model builders."""

import numpy as np
import pytest
from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error

from tn4qa.noise_model.noise_data import QubitNoise, NoiseData
from tn4qa.noise_model.device_characterisation import generate_noise_data
from tn4qa.noise_model.build import build_noise_inversion_channels, get_noise_model


@pytest.fixture
def noise_data() -> NoiseData:
    nqubits = 20
    coupling_map = [
        [0, 1],
        [1, 0],
        [0, 3],
        [3, 0],
        [1, 4],
        [4, 1],
        [2, 3],
        [3, 2],
        [2, 7],
        [7, 2],
        [3, 4],
        [4, 3],
        [3, 8],
        [8, 3],
        [4, 5],
        [5, 4],
        [4, 9],
        [9, 4],
        [5, 6],
        [6, 5],
        [5, 10],
        [10, 5],
        [6, 11],
        [11, 6],
        [7, 8],
        [8, 7],
        [7, 12],
        [12, 7],
        [8, 9],
        [9, 8],
        [8, 13],
        [13, 8],
        [9, 10],
        [10, 9],
        [9, 14],
        [14, 9],
        [10, 11],
        [11, 10],
        [10, 15],
        [15, 10],
        [11, 16],
        [16, 11],
        [12, 13],
        [13, 12],
        [13, 14],
        [14, 13],
        [13, 17],
        [17, 13],
        [14, 15],
        [15, 14],
        [14, 18],
        [18, 14],
        [15, 16],
        [16, 15],
        [15, 19],
        [19, 15],
        [17, 18],
        [18, 17],
        [18, 19],
        [19, 18],
    ]
    qubit_noise = [QubitNoise(id=0, t1_ns=43410.1770669608, t2_ns=5546.754503249878, readout_p0=0.9775, readout_p1=0.9775, gates_1q={'r': 0.9990007041848946}, gates_2q={1: 0.9902274607474589, 3: 0.9918029054337629}), 
                   QubitNoise(id=1, t1_ns=34968.13213140938, t2_ns=2575.3850545478826, readout_p0=0.9745, readout_p1=0.9745, gates_1q={'r': 0.9988573294753003}, gates_2q={0: 0.9902274607474589, 4: 0.9928324289462214}), 
                   QubitNoise(id=2, t1_ns=40758.22322522347, t2_ns=4755.037107857137, readout_p0=0.97025, readout_p1=0.97025, gates_1q={'r': 0.9989209203926381}, gates_2q={3: 0.9895114371586425, 7: 0.9924136510683637}), 
                   QubitNoise(id=3, t1_ns=42338.45323268269, t2_ns=1973.6215996647054, readout_p0=0.94575, readout_p1=0.94575, gates_1q={'r': 0.9982026563344413}, gates_2q={0: 0.9918029054337629, 2: 0.9895114371586425, 4: 0.9923078328485646, 8: 0.9911723862393091}), 
                   QubitNoise(id=4, t1_ns=31687.50089738787, t2_ns=6153.93566326493, readout_p0=0.9785, readout_p1=0.9785, gates_1q={'r': 0.9985737982361929}, gates_2q={1: 0.9928324289462214, 3: 0.9923078328485646, 5: 0.9923906053302095, 9: 0.9922840549492525}), 
                   QubitNoise(id=5, t1_ns=42426.53364451025, t2_ns=1975.0477210107958, readout_p0=0.96625, readout_p1=0.96625, gates_1q={'r': 0.997301056191781}, gates_2q={4: 0.9923906053302095, 6: 0.9928115849011512, 10: 0.9966063475177782}), 
                   QubitNoise(id=6, t1_ns=49154.75800608243, t2_ns=4524.78153423451, readout_p0=0.985, readout_p1=0.985, gates_1q={'r': 0.9990077404534111}, gates_2q={5: 0.9928115849011512, 11: 0.9930684645689096}), 
                   QubitNoise(id=7, t1_ns=45710.784280755084, t2_ns=2319.844285109669, readout_p0=0.982, readout_p1=0.982, gates_1q={'r': 0.998575519612879}, gates_2q={2: 0.9924136510683637, 8: 0.990161462057128, 12: 0.9925518010110758}), 
                   QubitNoise(id=8, t1_ns=34033.12093913222, t2_ns=5793.71725560215, readout_p0=0.9825, readout_p1=0.9825, gates_1q={'r': 0.9981202652492177}, gates_2q={3: 0.9911723862393091, 7: 0.990161462057128, 9: 0.9935597545907197, 13: 0.9910293936468578}), 
                   QubitNoise(id=9, t1_ns=34626.08700454571, t2_ns=4057.278698798379, readout_p0=0.9735, readout_p1=0.9735, gates_1q={'r': 0.9985727377738376}, gates_2q={4: 0.9922840549492525, 8: 0.9935597545907197, 10: 0.9905105109647068, 14: 0.9930580774748671}), 
                   QubitNoise(id=10, t1_ns=45313.26797672993, t2_ns=5129.240695206758, readout_p0=0.975, readout_p1=0.975, gates_1q={'r': 0.9990599188460332}, gates_2q={5: 0.9966063475177782, 9: 0.9905105109647068, 11: 0.9943033563599978, 15: 0.9947334779038436}), 
                   QubitNoise(id=11, t1_ns=31470.978839394556, t2_ns=3772.3398266344475, readout_p0=0.98575, readout_p1=0.98575, gates_1q={'r': 0.9988888285427157}, gates_2q={6: 0.9930684645689096, 10: 0.9943033563599978, 16: 0.9953867946344}), 
                   QubitNoise(id=12, t1_ns=25075.980459828505, t2_ns=10093.345670554701, readout_p0=0.9847499999999999, readout_p1=0.9847499999999999, gates_1q={'r': 0.9991012980913542}, gates_2q={7: 0.9925518010110758, 13: 0.992324869613737}), 
                   QubitNoise(id=13, t1_ns=41638.93148121549, t2_ns=1643.0198935823455, readout_p0=0.98125, readout_p1=0.98125, gates_1q={'r': 0.998639001154107}, gates_2q={8: 0.9910293936468578, 12: 0.992324869613737, 14: 0.9935454812885923, 17: 0.9916109430688291}), 
                   QubitNoise(id=14, t1_ns=41043.62523778416, t2_ns=6708.971051299335, readout_p0=0.978, readout_p1=0.978, gates_1q={'r': 0.9988358601558478}, gates_2q={9: 0.9930580774748671, 13: 0.9935454812885923, 15: 0.9945238371108084, 18: 0.9938364975254941}), 
                   QubitNoise(id=15, t1_ns=34900.5911723808, t2_ns=4335.6146733693295, readout_p0=0.9782499999999998, readout_p1=0.9782499999999998, gates_1q={'r': 0.9989017616812649}, gates_2q={10: 0.9947334779038436, 14: 0.9945238371108084, 16: 0.995059313567962, 19: 0.9928153664171769}), 
                   QubitNoise(id=16, t1_ns=49851.954491479424, t2_ns=9997.706131333185, readout_p0=0.9732499999999998, readout_p1=0.9732499999999998, gates_1q={'r': 0.9990831146886842}, gates_2q={11: 0.9953867946344, 15: 0.995059313567962}), 
                   QubitNoise(id=17, t1_ns=43652.37604385771, t2_ns=12076.604806666008, readout_p0=0.981, readout_p1=0.981, gates_1q={'r': 0.998870947755937}, gates_2q={13: 0.9916109430688291, 18: 0.9929779412150104}), 
                   QubitNoise(id=18, t1_ns=54674.48077842716, t2_ns=3962.871440200661, readout_p0=0.97125, readout_p1=0.97125, gates_1q={'r': 0.9991170876565165}, gates_2q={14: 0.9938364975254941, 17: 0.9929779412150104, 19: 0.9922477883919064}), 
                   QubitNoise(id=19, t1_ns=43505.55933919115, t2_ns=12679.178938060735, readout_p0=0.973, readout_p1=0.973, gates_1q={'r': 0.9994841361734668}, gates_2q={15: 0.9928153664171769, 18: 0.9922477883919064})]
    return NoiseData(nqubits, coupling_map, qubit_noise)

@pytest.fixture
def gate_duration_ns_1q() -> int:
    return 20


@pytest.fixture
def basis_gates_1q() -> list:
    return ["id", "r"]


@pytest.fixture
def basis_gates_2q() -> list:
    return ["cz"]


@pytest.fixture
def basis_gates(basis_gates_1q, basis_gates_2q) -> list:
    return basis_gates_1q + basis_gates_2q


def test_build_noise_inversion_channels(noise_data, gate_duration_ns_1q):
    noise_inversion_channels_1q, noise_inversion_channels_2q = (
        build_noise_inversion_channels(noise_data, gate_duration_ns_1q)
    )
    for i in range(noise_data.n_qubits):
        assert (
            str(i) in noise_inversion_channels_1q
        ), "Noise inversion channel should hold a dict for every qubit (in string format)."
        # readout error
        p0given0 = noise_data.qubit_noise[i].readout_p0
        p1given1 = noise_data.qubit_noise[i].readout_p1
        readout_mat = np.array([[p0given0, 1 - p0given0], [1 - p1given1, p1given1]])
        assert (
            "measurement" in noise_inversion_channels_1q[str(i)]
        ), "Noise inversion channel should have a 'measurement' entry for every qubit."
        assert np.allclose(
            noise_inversion_channels_1q[str(i)]["measurement"] @ readout_mat,
            np.identity(2),
        ), "Inversion channel should have inverse readout error."
        # incoherent error
        T1_ns = noise_data.qubit_noise[i].t1_ns
        T2_ns = noise_data.qubit_noise[i].t2_ns
        thermal_relaxation = thermal_relaxation_error(
            T1_ns * 10e-9, T2_ns * 10e-9, gate_duration_ns_1q * 10e-9
        )
        thermal_relaxation_mat = thermal_relaxation.to_quantumchannel().data
        assert (
            "thermal relaxation" in noise_inversion_channels_1q[str(i)]
        ), "Noise inversion channel should have a 'thermal relaxation' entry for every qubit."
        assert np.allclose(
            noise_inversion_channels_1q[str(i)]["thermal relaxation"]
            @ thermal_relaxation_mat,
            np.identity(4),
        ), "Inversion channel should have inverse thermal relaxation"
        # coherent error
        p_incoherent_error_1q = (gate_duration_ns_1q / T1_ns) + (
            gate_duration_ns_1q / T2_ns
        )
        p_coherent_error_1q = max(
            noise_data.qubit_noise[i].gates_1q["r"] - p_incoherent_error_1q, 0.0
        )
        if p_coherent_error_1q <= 0:
            assert (
                "coherent error" not in noise_inversion_channels_1q[str(i)]
            ), f"Qubit {i} has no coherent error."
        coherent_error_1q = depolarizing_error(p_coherent_error_1q, 1)
        coherent_error_1q_mat = coherent_error_1q.to_quantumchannel().data
        assert (
            "coherent error" in noise_inversion_channels_1q[str(i)]
        ), f"Qubit {i} needs coherent error entry."
        assert np.allclose(
            noise_inversion_channels_1q[str(i)]["coherent error"]
            @ coherent_error_1q_mat,
            np.identity(4),
        ), "Inversion channel should have inverse coherent error."
    for q1, q2 in noise_data.coupling_map:
        # two qubit error
        coherent_error_2q = depolarizing_error(
            noise_data.qubit_noise[q1].gates_2q[q2], 2
        )
        coherent_error_2q_mat = coherent_error_2q.to_quantumchannel().data
        assert (
            f"q{q1}q{q2}" in noise_inversion_channels_2q
        ), f"q{q1}q{q2} in coupling map but not in noise inversion channel."
        assert np.allclose(
            noise_inversion_channels_2q[f"q{q1}q{q2}"] @ coherent_error_2q_mat,
            np.identity(16),
        ), "Inversion channel should have inverse two-qubit error."

    return


def test_noise_model(noise_data, gate_duration_ns_1q, basis_gates):
    noise_model = get_noise_model(noise_data, basis_gates, gate_duration_ns_1q)
    assert set(noise_model.basis_gates) == set(
        basis_gates
    ), "Basis gates set must correspond to the one given in input."
    assert set(noise_model.noise_instructions) == set(
        basis_gates + ["measure"]
    ), "Instructions with noise are all the basis set gates + measurement."
    assert noise_model.noise_qubits == list(
        range(noise_data.n_qubits)
    ), f"All qubits from 0 to {noise_data.n_qubits} must be regarded as noisy."
    noise_qubits_list = noise_model.to_dict()["errors"]
    # FIXME: we don't always have coherent error on 'r'!
    # TODO: Test should be generalized to different basis gates for different devices
    assert isinstance(noise_qubits_list, list) and len(
        noise_qubits_list
    ) == noise_data.n_qubits * 3 + len(
        noise_data.coupling_map
    ), "Noise model should have one dict for every qubit and every 1-qubit error + measurement or for every couple of interacting qubits."
    for error_dict in noise_qubits_list:
        assert any(
            [error_dict["operations"] == [gate] for gate in basis_gates + ["measure"]]
        ), "Error model for a gate not present in the basis gates list (+ measurement)."
        if len(error_dict["gate_qubits"][0]) == 1:
            assert (
                0 <= error_dict["gate_qubits"][0][0] < noise_data.n_qubits
            ), "Qubit number out of range."
        else:
            assert (
                list(error_dict["gate_qubits"][0]) in noise_data.coupling_map
            ), "Couple of qubits not connected on the device."

    return


