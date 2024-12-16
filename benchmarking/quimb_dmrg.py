import time

import quimb.tensor as qtn
from quimb.tensor.tensor_dmrg import DMRG, DMRG2


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


# Different Hamiltonians for variety


def create_hamiltonians(n_sites):
    """Create a variety of MPO Hamiltonians for testing."""
    J = 1.0
    delta = 0.5  # XXZ anisotropy

    # Heisenberg Hamiltonian

    heis_ham = qtn.SpinHam1D(S=1 / 2)
    heis_ham += J, "X", "X"
    heis_ham += J, "Y", "Y"
    heis_ham += J, "Z", "Z"
    heis_mpo = heis_ham.build_mpo(n_sites)

    # XXZ Hamiltonian

    xxz_ham = qtn.SpinHam1D(S=1 / 2)
    xxz_ham += J, "X", "X"
    xxz_ham += J, "Y", "Y"
    xxz_ham += delta * J, "Z", "Z"
    xxz_mpo = xxz_ham.build_mpo(n_sites)

    return {"Heisenberg": heis_mpo, "XXZ": xxz_mpo}


# Run benchmarks

results = []
for n_sites in system_sizes:
    hamiltonians = create_hamiltonians(n_sites)
    for h_name, hamiltonian in hamiltonians.items():
        for bond_dims in bond_dims_list:
            for tol in tolerances:
                for max_sweeps in max_sweeps_list:
                    for method in methods:
                        result = run_dmrg_benchmark(
                            method, hamiltonian, bond_dims, tol, max_sweeps
                        )
                        result.update(
                            {
                                "hamiltonian": h_name,
                                "n_sites": n_sites,
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
        f"System size: {result['n_sites']} sites\n"
        f"Bond dimensions: {result['bond_dims']}\n"
        f"Tolerance: {result['tol']}\n"
        f"Max sweeps: {result['max_sweeps']}\n"
        f"Method: {result['method']}\n"
        f"  Converged: {result['converged']}\n"
        f"  Ground state energy: {result['energy']}\n"
        f"  Runtime: {result['runtime']:.2f} seconds\n"
    )
