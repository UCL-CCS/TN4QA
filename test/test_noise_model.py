from tn4qa.noise_model.device_characterisation import get_coupling_map, generate_noise_data
from tn4qa.noise_model.build import build_noise_inversion_channels

import pytest
import json
import numpy as np
from collections import Counter
from qiskit_aer.noise import thermal_relaxation_error, depolarizing_error

nqubits = 20
with open("test/data/qexa-calibration-data-2024-10-15.json", "r") as f:
    data = json.load(f)
gate_duration_ns_1q = 20


def test_get_coupling_map():
    coupling_map = get_coupling_map(nqubits, data)
    for couple in coupling_map:
        assert len(couple) == 2, "Coupling map should be a list of pairs of connected qubits."
        assert 0 <= couple[0] < nqubits and 0 <= couple[1] < nqubits, "The qubit indices should run from 0 to nqubits -1."
        assert [couple[1], couple[0]] in coupling_map, "The coupling map should be symmetric."
    
    return


def test_build_noise_data():
    noise_data = generate_noise_data(nqubits, data)
    for i in range(nqubits):
        assert str(i) in noise_data, "Noise data should hold a dict for every qubit (in string format)."
        assert "T1_ns" in noise_data[str(i)], "Every qubit should have 'T1_ns' field."
        assert isinstance(noise_data[str(i)]["T1_ns"], float), "T1_ns should be a float."
        assert "T2_ns" in noise_data[str(i)], "Every qubit should have 'T2_ns' field."
        assert isinstance(noise_data[str(i)]["T2_ns"], float), "T2_ns should be a float."
        assert "measurements" in noise_data[str(i)], "Every qubit should have 'measurements' field."
        assert "p00" in noise_data[str(i)]["measurements"], "field 'measurements' should have field 'p00'."
        assert isinstance(noise_data[str(i)]["measurements"]["p00"], float) and \
               0 <= noise_data[str(i)]["measurements"]["p00"] <=1, "p00 should be a float between 0 and 1."
        assert "p11" in noise_data[str(i)]["measurements"], "field 'measurements' should have field 'p11'."
        assert isinstance(noise_data[str(i)]["measurements"]["p11"], float) and \
               0 <= noise_data[str(i)]["measurements"]["p11"] <=1, "p11 should be a float between 0 and 1."
        assert "gates_1q" in noise_data[str(i)], "Every qubit should have 'gates_1q' field."
        assert isinstance(noise_data[str(i)]["gates_1q"], dict) and "r" in noise_data[str(i)]["gates_1q"], "1 qubit gates entry needs 'r' field."
        assert isinstance(noise_data[str(i)]["gates_1q"]["r"], float) and \
               0 <= noise_data[str(i)]["gates_1q"]["r"] <= 1, "'r' should be a float between 0 and 1."
        assert "gates_2q" in noise_data[str(i)], "Every qubit should have 'gates_2q' field."
        assert isinstance(noise_data[str(i)]["gates_2q"], dict), "2 qubit gates field must be a dict with one entry for each connected qubit."
        for key, value in noise_data[str(i)]["gates_2q"].items():
            assert isinstance(key, str) and 0 <= int(key) < nqubits, "dict of 2 qubit gates must have the number of connected qubit as key (str)."
            assert isinstance(value, float) and 0 <= value <= 1, "2 qubit gates fidelity should be a float between 0 and 1."

    return


def test_match_coupling_noise_data():
    coupling_map = get_coupling_map(nqubits, data)
    noise_data = generate_noise_data(nqubits, data)
    map_from_noise_data = []
    for qubit, qubit_dict in noise_data.items():
        for second_qubit in qubit_dict["gates_2q"]:
            assert [int(qubit), int(second_qubit)] in coupling_map, f"Couple of qubits ({qubit}, {second_qubit}) in noise_data but not in coupling_map."
    for couple in coupling_map:
        assert str(couple[0]) in noise_data, f"Qubit {couple[0]} in coupling_map but not in noise_data."
        assert str(couple[1]) in noise_data[str(couple[0])]["gates_2q"], f"Couple of qubits ({couple[0],couple[1]}) in coupling_map but not noise_data"
    
    return


def test_build_noise_inversion_channels():
    noise_data = generate_noise_data(nqubits, data)
    noise_inversion_channels_1q, noise_inversion_channels_2q = build_noise_inversion_channels(nqubits, data, gate_duration_ns_1q)
    coupling_map = get_coupling_map(nqubits, data)
    for i in range(nqubits):
        assert str(i) in noise_inversion_channels_1q, "Noise inversion channel should hold a dict for every qubit (in string format)."
        # readout error
        p0given0 = noise_data[str(i)]["measurements"]["p00"]
        p1given1 = noise_data[str(i)]["measurements"]["p11"]
        readout_mat = np.array([[p0given0, 1 - p0given0], [1 - p1given1, p1given1]])
        assert "measurement" in noise_inversion_channels_1q[str(i)], "Noise inversion channel should have a 'measurement' entry for every qubit."
        assert np.allclose(noise_inversion_channels_1q[str(i)]["measurement"]@readout_mat, np.identity(2)), "Inversion channel should have inverse readout error."
        # incoherent error
        T1_ns = noise_data[str(i)]["T1_ns"]
        T2_ns = noise_data[str(i)]["T2_ns"]
        thermal_relaxation = thermal_relaxation_error(
            T1_ns * 10e-9, T2_ns * 10e-9, gate_duration_ns_1q * 10e-9
        )
        thermal_relaxation_mat = thermal_relaxation.to_quantumchannel().data
        assert "thermal relaxation" in noise_inversion_channels_1q[str(i)], "Noise inversion channel should have a 'thermal relaxation' entry for every qubit."
        assert np.allclose(noise_inversion_channels_1q[str(i)]["thermal relaxation"]@thermal_relaxation_mat, np.identity(4)), "Inversion channel should have inverse thermal relaxation"
        # coherent error
        p_incoherent_error_1q = (gate_duration_ns_1q / T1_ns) + (gate_duration_ns_1q / T2_ns)
        p_coherent_error_1q = max(noise_data[str(i)]["gates_1q"]["r"] - p_incoherent_error_1q, 0.0)
        if p_coherent_error_1q <= 0:
            assert "coherent error" not in noise_inversion_channels_1q[str(i)], f"Qubit {i} has no coherent error."
        coherent_error_1q = depolarizing_error(p_coherent_error_1q, 1)
        coherent_error_1q_mat = coherent_error_1q.to_quantumchannel().data
        assert "coherent error" in noise_inversion_channels_1q[str(i)], f"Qubit {i} needs coherent error entry."
        assert np.allclose(noise_inversion_channels_1q[str(i)]["coherent error"]@coherent_error_1q_mat, np.identity(4)), "Inversion channel should have inverse coherent error."
    for q1, q2 in coupling_map:
        # two qubit error
        coherent_error_2q = depolarizing_error(noise_data[str(q1)]["gates_2q"][str(q2)], 2)
        coherent_error_2q_mat = coherent_error_2q.to_quantumchannel().data
        assert f"q{q1}q{q2}" in noise_inversion_channels_2q, f"q{q1}q{q2} in coupling map but not in noise inversion channel."
        assert np.allclose(noise_inversion_channels_2q[f"q{q1}q{q2}"]@coherent_error_2q_mat, np.identity(16)), "Inversion channel should have inverse two-qubit error."
    
    return

if __name__ == "__main__":
    test_get_coupling_map()
    test_build_noise_data()
    test_match_coupling_noise_data()
    test_build_noise_inversion_channels()






    