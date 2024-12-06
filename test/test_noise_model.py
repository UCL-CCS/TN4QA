from tn4qa.noise_model.device_characterisation import get_coupling_map, generate_noise_data

import pytest
import json
import numpy as np

nqubits = 20
with open("test/data/qexa-calibration-data-2024-10-15.json", "r") as f:
    data = json.load(f)

def test_get_coupling_map():
    coupling_map = get_coupling_map(nqubits, data)
    for couple in coupling_map:
        assert len(couple) == 2, "Coupling map should be a list of pairs of connected qubits."
        assert 0 <= couple[0] < nqubits and 0 <= couple[1] < nqubits, "The qubit indices should run from 0 to nqubits -1."
        assert [couple[1], couple[0]] in coupling_map, "The coupling map should be symmetric."


def test_build_noise_data():
    noise_data = generate_noise_data(nqubits, data)
    for i in range(nqubits):
        assert str(i) in noise_data, "Noise data should hold a dict for every qubit (in string format)."
        assert 'T1_ns' in noise_data[str(i)], "Every qubit should have 'T1_ns' field."
        assert 'T2_ns' in noise_data[str(i)], "Every qubit should have 'T2_ns' field."
        assert 'measurements' in noise_data[str(i)], "Every qubit should have 'measurements' field."
        assert 'gates_1q' in noise_data[str(i)], "Every qubit should have 'gates_1q' field."
        assert 'gates_2q' in noise_data[str(i)], "Every qubit should have 'gates_2q' field."

    print(noise_data)



if __name__ == "__main__":
    test_get_coupling_map()
    test_build_noise_data()






    