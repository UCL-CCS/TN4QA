"""Test for the device characterisation."""

import json
import pytest

from tn4qa.noise_model.device_characterisation import (
    QubitNoise,
    generate_noise_data,
    get_coupling_map,
)


@pytest.fixture
def calibration_data() -> dict:
    with open("test/data/qexa-calibration-data-2024-10-15.json", "r") as f:
        data = json.load(f)
    return data


@pytest.fixture
def nqubits() -> int:
    return 20


def test_get_coupling_map(nqubits, calibration_data):
    coupling_map = get_coupling_map(nqubits, calibration_data)
    for couple in coupling_map:
        assert (
            len(couple) == 2
        ), "Coupling map should be a list of pairs of connected qubits."
        assert (
            0 <= couple[0] < nqubits and 0 <= couple[1] < nqubits
        ), "The qubit indices should run from 0 to nqubits -1."
        assert [
            couple[1],
            couple[0],
        ] in coupling_map, "The coupling map should be symmetric."

    return


def test_build_noise_data(nqubits, calibration_data):
    noise_data = generate_noise_data(nqubits, calibration_data)
    assert (
        noise_data.n_qubits == nqubits
    ), f"The field n_qubits should be equal to {nqubits}."
    assert isinstance(
        noise_data.qubit_noise, list
    ), "NoiseData.qubit_noise should be a list of QubitNoise instances."
    assert (
        len(noise_data.qubit_noise) == nqubits
    ), "There should be one QubitNoise instance for evert qubit."
    for i in range(nqubits):
        assert isinstance(
            noise_data.qubit_noise[i], QubitNoise
        ), "There should be one QubitNoise for every qubit."
        assert isinstance(
            noise_data.qubit_noise[i].t1_ns, float
        ), "t1_ns should be a float."
        assert isinstance(
            noise_data.qubit_noise[i].t2_ns, float
        ), "t2_ns should be a float."
        assert (
            isinstance(noise_data.qubit_noise[i].readout_p0, float)
            and 0 <= noise_data.qubit_noise[i].readout_p0 <= 1
        ), "readout_p0 should be a float between 0 and 1."
        assert (
            isinstance(noise_data.qubit_noise[i].readout_p1, float)
            and 0 <= noise_data.qubit_noise[i].readout_p1 <= 1
        ), "readout_p1 should be a float between 0 and 1."
        assert isinstance(
            noise_data.qubit_noise[i].gates_1q, dict
        ), "gates_1q should be a dict."
        assert "r" in noise_data.qubit_noise[i].gates_1q, "key 'r' in gates_1q missing."
        assert (
            isinstance(noise_data.qubit_noise[i].gates_1q["r"], float)
            and 0 <= noise_data.qubit_noise[i].gates_1q["r"] <= 1
        ), "'r' should be a float between 0 and 1."
        assert isinstance(
            noise_data.qubit_noise[i].gates_2q, dict
        ), "2-qubit gates field must be a dict."
        for key, value in noise_data.qubit_noise[i].gates_2q.items():
            assert (
                isinstance(key, str) and 0 <= int(key) < nqubits
            ), "dict of 2-qubit gates must have the number of connected qubit as key (str)."
            assert (
                isinstance(value, float) and 0 <= value <= 1
            ), "2-qubit gates fidelity should be a float between 0 and 1."

    return


def test_match_coupling_noise_data(nqubits, calibration_data):
    noise_data = generate_noise_data(nqubits, calibration_data)
    for qubit, single_qubit_noise in enumerate(noise_data.qubit_noise):
        for second_qubit in single_qubit_noise.gates_2q.keys():
            assert [
                qubit,
                second_qubit,
            ] in noise_data.coupling_map, f"Couple of qubits ({qubit}, {second_qubit}) in noise_data but not in coupling_map."
    for couple in noise_data.coupling_map:
        assert (
            0 <= couple[0] < noise_data.n_qubits
        ), "Qubit in coupling_map out of range."
        assert (
            0 <= couple[1] < noise_data.n_qubits
        ), "Qubit in coupling_map out of range."
        assert (
            couple[1] in noise_data.qubit_noise[couple[0]].gates_2q
        ), f"Couple of qubits ({couple[0],couple[1]}) in coupling_map but not noise_data"
        assert (
            couple[0] in noise_data.qubit_noise[couple[1]].gates_2q
        ), f"Couple of qubits ({couple[1],couple[0]}) in coupling_map but not noise_data"
    return
