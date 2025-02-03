import json
import os
import time
import numpy as np

from quimb.tensor.tensor_dmrg import DMRG, DMRG2
from tn4qa.dmrg import FermionDMRG, QubitDMRG

from benchmarking.utils import _hamiltonian_to_mpo, load_scf_from_chk
from pyscf import scf

def load_hamiltonian(ham_file_path):
    """Load the Hamiltonian from a JSON file"""
    with open(ham_file_path, "r") as f:
        data = json.load(f)

    return data

def create_mpo(ham_file_path):
    """Convert Hamiltonian to an MPO using the provided helper function."""
    hamiltonian = load_hamiltonian(ham_file_path)
    print(f"Loaded Hamiltonian from {ham_file_path}")

    return _hamiltonian_to_mpo(hamiltonian), hamiltonian

def run_dmrg_benchmark(method, ham_file_path, scf_file_path, bond_dims, tol, max_sweeps):
    """Run a DMRG method and return results."""
    start_time = time.time()

    # Load Hamiltonian & MPO
    mpo, hamiltonian = create_mpo(ham_file_path)
    ham_dict = {k: float(v[0]) for k, v in hamiltonian.items()}

    # Select the appropriate DMRG solver
    if method == "DMRG":
        dmrg_solver = DMRG(mpo, bond_dims=bond_dims, cutoffs=1e-8)
        converged = dmrg_solver.solve(tol=tol, max_sweeps=max_sweeps, verbosity=0)
        energy = dmrg_solver.energy
    elif method == "DMRG2":
        dmrg_solver = DMRG2(mpo, bond_dims=bond_dims, cutoffs=1e-8)
        converged = dmrg_solver.solve(tol=tol, max_sweeps=max_sweeps, verbosity=0)
        energy = dmrg_solver.energy
    elif method == "FermionDMRG":
        # Use helper function to prepare the SCF object
        mf = load_scf_from_chk(scf_file_path, "RHF")

        # Initialise FermionDMRG with the SCF object
        dmrg_solver = FermionDMRG(scf_obj=mf, HF_symmetry="RHF", max_mps_bond=bond_dims[-1])

        energy = dmrg_solver.run(maxiter=max_sweeps)
    elif method == "QubitDMRG":
        dmrg_solver = QubitDMRG(ham_dict, max_mps_bond=bond_dims[-1])
        energy, _ = dmrg_solver.run(maxiter=max_sweeps)
    else:
        raise ValueError(f"Unknown method: {method}")

    end_time = time.time()

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
methods = ["DMRG", "DMRG2", "FermionDMRG", "QubitDMRG"]  

# Path to the Hamiltonian
ham_file_paths= ["hamiltonians/HeH.json", "hamiltonians/LiH.json"]
scf_file_paths = ["hamiltonians/HeH_sto-3g.chk", "hamiltonians/LiH_sto-3g.chk"]
file_paths = [["hamiltonians/HeH.json","hamiltonians/HeH_sto-3g.chk"],["hamiltonians/LiH.json", "hamiltonians/LiH_sto-3g.chk"]]
# Run benchmarks

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
                                "hamiltonian": os.path.splitext(os.path.basename(ham_file_path))[0],
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
