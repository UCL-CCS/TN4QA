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
    
    return


def test_build_noise_data():
    noise_data = generate_noise_data(nqubits, data)
    for i in range(nqubits):
        assert str(i) in noise_data, "Noise data should hold a dict for every qubit (in string format)."
        assert 'T1_ns' in noise_data[str(i)], "Every qubit should have 'T1_ns' field."
        assert isinstance(noise_data[str(i)]['T1_ns'], float), "T1_ns should be a float."
        assert 'T2_ns' in noise_data[str(i)], "Every qubit should have 'T2_ns' field."
        assert isinstance(noise_data[str(i)]['T2_ns'], float), "T2_ns should be a float."
        assert 'measurements' in noise_data[str(i)], "Every qubit should have 'measurements' field."
        assert 'p00' in noise_data[str(i)]['measurements'], "field 'measurements' should have field 'p00'."
        assert isinstance(noise_data[str(i)]['measurements']['p00'], float) and \
               0 <= noise_data[str(i)]['measurements']['p00'] <=1, "p00 should be a float between 0 and 1."
        assert 'p11' in noise_data[str(i)]['measurements'], "field 'measurements' should have field 'p11'."
        assert isinstance(noise_data[str(i)]['measurements']['p11'], float) and \
               0 <= noise_data[str(i)]['measurements']['p11'] <=1, "p11 should be a float between 0 and 1."
        assert 'gates_1q' in noise_data[str(i)], "Every qubit should have 'gates_1q' field."
        assert isinstance(noise_data[str(i)]['gates_1q'], dict) and 'r' in noise_data[str(i)]['gates_1q'], "1 qubit gates entry needs 'r' field."
        assert isinstance(noise_data[str(i)]['gates_1q']['r'], float) and \
               0 <= noise_data[str(i)]['gates_1q']['r'] <= 1, "r should be a float between 0 and 1."
        assert 'gates_2q' in noise_data[str(i)], "Every qubit should have 'gates_2q' field."
        assert isinstance(noise_data[str(i)]['gates_2q'], dict), "2 qubit gates field must be a dict with one entry for each connected qubit."
        for key, value in noise_data[str(i)]['gates_2q'].items():
            assert isinstance(key, str) and 0 <= int(key) < nqubits, "dict of 2 qubit gates must have the number of connected qubit as key (str)."
            assert isinstance(value, float) and 0 <= value <= 1, "2 qubit gates fidelity should be a float between 0 and 1."

    return





if __name__ == "__main__":
    test_get_coupling_map()
    test_build_noise_data()






    