import json
import os
import time

from quimb.tensor.tensor_dmrg import DMRG, DMRG2

from qmmbed.tense import _hamiltonian_to_mpo


def load_hamiltonian(file_path):
    """Load the Hamiltonian from a JSON file and convert to QubitOperator."""
    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def create_mpo(file_path):
    """Convert Hamiltonian to an MPO using the provided helper function."""
    hamiltonian = load_hamiltonian(file_path)
    print(hamiltonian)

    return _hamiltonian_to_mpo(hamiltonian)


def run_dmrg_benchmark(method, ham, bond_dims, tol, max_sweeps):
    """Run a DMRG method and return results."""
    start_time = time.time()

    if method == "DMRG":
        dmrg_solver = DMRG(ham, bond_dims=bond_dims, cutoffs=1e-8)
    elif method == "DMRG2":
        dmrg_solver = DMRG2(ham, bond_dims=bond_dims, cutoffs=1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    converged = dmrg_solver.solve(tol=tol, max_sweeps=max_sweeps, verbosity=0)
    end_time = time.time()

    return {
        "method": method,
        "converged": converged,
        "energy": dmrg_solver.energy,
        "runtime": end_time - start_time,
    }


# Benchmark settings

system_sizes = [10, 20]
bond_dims_list = [[8, 16, 32], [16, 32, 64]]
tolerances = [1e-4, 1e-6]
max_sweeps_list = [10, 20]
methods = ["DMRG", "DMRG2"]

# Create MPO
file_path = "C:/Users/isabe/OneDrive/Documenti/isabe/PycharmProjects/QETAMINE/hamiltonians/LiH.json"
mpo = create_mpo(file_path)


# Run benchmarks

results = []
for bond_dims in bond_dims_list:
    for tol in tolerances:
        for max_sweeps in max_sweeps_list:
            for method in methods:
                result = run_dmrg_benchmark(
                    method, mpo, bond_dims, tol, max_sweeps
                )
                result.update(
                    {
                        "hamiltonian": os.path.splitext(os.path.basename(file_path))[0],
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
        f"  Converged: {result['converged']}\n"
        f"  Ground state energy: {result['energy']}\n"
        f"  Runtime: {result['runtime']:.2f} seconds\n"
    )

