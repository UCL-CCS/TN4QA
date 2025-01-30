import json
import os
import time

from quimb.tensor.tensor_dmrg import DMRG, DMRG2
from tn4qa.dmrg import FermionDMRG, QubitDMRG

from benchmarking.utils import _hamiltonian_to_mpo
from pyscf import gto, scf


def load_hamiltonian(file_path):
    """Load the Hamiltonian from a JSON file"""
    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def create_mpo(file_path):
    """Convert Hamiltonian to an MPO using the provided helper function."""
    hamiltonian = load_hamiltonian(file_path)
    print(hamiltonian)

    return _hamiltonian_to_mpo(hamiltonian)


def run_dmrg_benchmark(method, file_path, bond_dims, tol, max_sweeps):
    """Run a DMRG method and return results."""
    start_time = time.time()

    # Load Hamiltonian and create MPO (not used for FermionDMRG directly)
    mpo = create_mpo(file_path)

    # Prepare the SCF object for FermionDMRG
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
    if "UHF" in method:
        mf = scf.UHF(mol).run(conv_tol=1e-14)
    else:
        mf = scf.RHF(mol).run(conv_tol=1e-14)

    # Initialise the appropriate DMRG solver
    if method == "DMRG":
        dmrg_solver = DMRG(mpo, bond_dims=bond_dims, cutoffs=1e-8)
    elif method == "DMRG2":
        dmrg_solver = DMRG2(mpo, bond_dims=bond_dims, cutoffs=1e-8)
    elif method == "FermionDMRG":
        # For FermionDMRG, we pass the SCF object (mf) and not the MPO
        dmrg_solver = FermionDMRG(scf_obj=mf, HF_symmetry="RHF", max_mps_bond=bond_dims[-1])
        energy = dmrg_solver.run(maxiter=max_sweeps)
        converged = True  # FermionDMRG doesn't have a 'solve' method
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # For other methods, we solve the DMRG
    if method != "FermionDMRG":
        converged = dmrg_solver.solve(tol=tol, max_sweeps=max_sweeps, verbosity=0)
        energy = dmrg_solver.energy
        
        # Check if the DMRG converged successfully before accessing the energy
        if not converged:
            raise RuntimeError(f"DMRG did not converge for method {method}.")
    end_time = time.time()

    # Return the results in a consistent format
    return {
        "method": method,
        "converged": converged,
        "energy": energy,
        "runtime": end_time - start_time,
    }


# Benchmark settings

system_sizes = [10, 20]
bond_dims_list = [[8, 16, 32], [16, 32, 64]]
tolerances = [1e-4, 1e-6]
max_sweeps_list = [10, 20]
methods = ["DMRG", "DMRG2", "FermionDMRG"]  # Added FermionDMRG to the methods list

# Path to the Hamiltonian
file_path = "/workspaces/TN4QA/hamiltonians/LiH.json"
mpo = create_mpo(file_path)

# Run benchmarks

results = []
for bond_dims in bond_dims_list:
    for tol in tolerances:
        for max_sweeps in max_sweeps_list:
            for method in methods:
                result = run_dmrg_benchmark(
                    method, file_path, bond_dims, tol, max_sweeps
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
