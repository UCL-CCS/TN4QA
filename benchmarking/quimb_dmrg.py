import json
import os
import time

from quimb.tensor.tensor_dmrg import DMRG2

from benchmarking.utils import _hamiltonian_to_mpo
from tn4qa.dmrg import DMRG


def load_hamiltonian(ham_file_path):
    """Load the Hamiltonian from a JSON file"""
    with open(ham_file_path) as f:
        data = json.load(f)

    return data


def create_mpo(ham_file_path):
    """Convert Hamiltonian to an MPO using the provided helper function."""
    hamiltonian = load_hamiltonian(ham_file_path)
    print(f"Loaded Hamiltonian from {ham_file_path}")

    return _hamiltonian_to_mpo(hamiltonian), hamiltonian


def run_dmrg_benchmark(
    method, ham_file_path, scf_file_path, bond_dims, tol, max_sweeps
):
    """Run a DMRG method and return results."""

    # Load Hamiltonian & MPO
    mpo, hamiltonian = create_mpo(ham_file_path)
    ham_dict = {k: float(v[0]) for k, v in hamiltonian.items()}

    # Select the appropriate DMRG solver
    if method == "DMRG2":
        dmrg_solver = DMRG2(mpo, bond_dims=bond_dims, cutoffs=1e-8)
        start_time = time.time()
        _ = dmrg_solver.solve(tol=tol, max_sweeps=max_sweeps, verbosity=0)
        end_time = time.time()
        energy = dmrg_solver.energy
    elif method == "DMRG":
        dmrg_solver = DMRG(ham_dict, max_mps_bond=bond_dims[-1], method="two-site")
        start_time = time.time()
        energy, _ = dmrg_solver.run(maxiter=max_sweeps)
        end_time = time.time()
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "method": method,
        "hamiltonian": os.path.splitext(os.path.basename(ham_file_path))[0],
        "bond_dims": bond_dims,
        "tol": tol,
        "max_sweeps": max_sweeps,
        "energy": energy,
        "runtime": end_time - start_time,
    }


# Benchmark settings

system_sizes = 10
bond_dims_list = [[8, 16, 32]]
tolerances = [1e-4]
max_sweeps_list = [4]
methods = ["DMRG", "DMRG2"]

# Path to the Hamiltonian
ham_file_paths = [
    "molecules/hamiltonians/sto_3g/HeH.json",
    "molecules/hamiltonians/sto_3g/LiH.json",
    "molecules/hamiltonians/sto_3g/water.json",
]
scf_file_paths = [
    "molecules/scf/sto_3g/HeH.chk",
    "molecules/scf/sto_3g/LiH.chk",
    "molecules/scf/sto_3g/water.chk",
]
file_paths = [
    [ham_file_paths[i], scf_file_paths[i]] for i in range(len(ham_file_paths))
]

results = []
for ham_file_path, scf_file_path in file_paths:
    for bond_dims in bond_dims_list:
        for tol in tolerances:
            for max_sweeps in max_sweeps_list:
                for method in methods:
                    result = run_dmrg_benchmark(
                        method, ham_file_path, scf_file_path, bond_dims, tol, max_sweeps
                    )
                    result.update(
                        {
                            "hamiltonian": os.path.splitext(
                                os.path.basename(ham_file_path)
                            )[0],
                            "bond_dims": bond_dims,
                            "tol": tol,
                            "max_sweeps": max_sweeps,
                        }
                    )
                    results.append(result)

# Output benchmarking results

for result in results:
    print(
        f"Hamiltonian: {result['hamiltonian']}\n"
        f"Bond dimensions: {result['bond_dims']}\n"
        f"Tolerance: {result['tol']}\n"
        f"Max sweeps: {result['max_sweeps']}\n"
        f"Method: {result['method']}\n"
        f"  Ground state energy: {result['energy']}\n"
        f"  Runtime: {result['runtime']:.2f} seconds\n"
    )
