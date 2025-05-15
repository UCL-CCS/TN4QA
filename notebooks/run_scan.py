import numpy as np
import pyscf
from tn4qa.dmrg import DMRG
from tn4qa.qi_metrics import get_all_mutual_information
from tn4qa.utils import ham_dict_from_scf
import json
from qiskit import QuantumCircuit

Rs = np.linspace(0.8, 3.0, 23)

results = {}



for R in Rs:
    R_str = str(round(R, 1))
    results[R_str] = {}

    mol = pyscf.M(
        atom='N 0 0 0; N 0 0 {}'.format(R),
        basis='STO-3G',
    )
    mf = pyscf.scf.RHF(mol).run(verbose=0)

    norb = mol.nao
    nocc = np.count_nonzero(mf.mo_occ)
    nvir = mol.nao - np.count_nonzero(mf.mo_occ)

    mycc = pyscf.cc.CCSD(mf).run(verbose=0)
    myfci = pyscf.fci.FCI(mf).run(verbose=0)

    results[R_str]['norb'] = norb
    results[R_str]['nocc'] = nocc
    results[R_str]['nvir'] = nvir

    results[R_str]['E_RHF'] = mf.e_tot
    results[R_str]['E_CCSD'] = mycc.e_tot
    results[R_str]['E_FCI'] = myfci.e_tot

    results[R_str]['mo_energy_RHF'] = mf.mo_energy.tolist()

    ham = ham_dict_from_scf(mf, qubit_transformation="JW")
    dmrg = DMRG(
        hamiltonian=ham,
        max_mps_bond=16,
        method="one-site"
    )
    dmrg.run(10)
    print(dmrg.energy)
    results[R_str]['E_DMRG'] = dmrg.energy

    mi_matrix = get_all_mutual_information(dmrg.mps)

    results[R_str]['mi_matrix'] = mi_matrix.tolist()

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
