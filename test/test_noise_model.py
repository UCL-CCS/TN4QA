from tn4qa.noise_model.device_characterisation import get_coupling_map

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




if __name__ == "__main__":
    test_get_coupling_map()







    