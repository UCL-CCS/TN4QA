import json
import os
from timeit import default_timer

from tn4qa.dmrg import QubitDMRG

cwd = os.getcwd()


def test_timings():
    times = []
    energies = []

    start = default_timer()
    location = os.path.join(cwd, "hamiltonians/HeH.json")
    with open(location) as f:
        ham = json.load(f)
    ham_dict = {k: float(v[0]) for k, v in ham.items()}
    dmrg = QubitDMRG(ham_dict, 4)
    energy, _ = dmrg.run(2)
    energies.append(energy)
    stop1 = default_timer()
    times.append(stop1 - start)

    location = os.path.join(cwd, "hamiltonians/LiH.json")
    with open(location) as f:
        ham = json.load(f)
    ham_dict = {k: float(v[0]) for k, v in ham.items()}
    dmrg = QubitDMRG(ham_dict, 4)
    energy, _ = dmrg.run(2)
    energies.append(energy)
    stop2 = default_timer()
    times.append(stop2 - stop1)

    location = os.path.join(cwd, "hamiltonians/N2.json")
    with open(location) as f:
        ham = json.load(f)
    ham_dict = {k: float(v) for k, v in ham.items()}
    dmrg = QubitDMRG(ham_dict, 4, "subspace-expansion")
    energy, _ = dmrg.run(2)
    energies.append(energy)
    stop3 = default_timer()
    times.append(stop3 - stop1)

    dict = {
        "HeH": {"Time": times[0], "Energy": energies[0]},
        "LiH": {"Time": times[1], "Energy": energies[1]},
        "N2": {"Time": times[2], "Energy": energies[2]},
    }

    with open("two_site_method.json", "w") as f:
        json.dump(dict, f)
