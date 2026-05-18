import os
import json
import numpy as np
from tn4qa.mps import MatrixProductState
from tn4qa.dmrg import DMRG
from tn4qa.tn_methods.active_space_selection import autocas_selection_ranked_entropy_threshold, autocas_selection_ranked_fixed_number, autocas_selection_total_entropy

cwd = os.getcwd()
mol_info_dir = os.path.join("molecules/mol_info")
mol_list = sorted(os.listdir(mol_info_dir))

mol_idx = 0
filepath = os.path.join(mol_info_dir, mol_list[mol_idx])

with open(filepath, "r") as f:
    info = json.load(f)

mol_name = info["name"]
mol_string = info["xyz"]
mol_charge = info["charge"]

ham_dir = os.path.join("molecules/hamiltonians")
filename = f"{mol_name}_STO-3G_RHF.json" 
filepath = os.path.join(ham_dir, filename)
with open(filepath, "r") as f:
    ham = json.load(f)


hf_mps = MatrixProductState.from_hf_state(16, 8)


# Run cheap DMRG to get MPS for testing
dmrg = DMRG(hamiltonian=ham, max_mps_bond=2, initial_mps=hf_mps)
_, mps = dmrg.run(5)


assert isinstance(mps, MatrixProductState)

top_orbitals = autocas_selection_ranked_fixed_number(mps, n_sites=16, active_orbitals=4)
print("Top orbitals (fixed number):", top_orbitals)
