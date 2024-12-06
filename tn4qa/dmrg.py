from typing import Tuple 
from tn4qa.mps import MatrixProductState
from tn4qa.mpo import MatrixProductOperator
from tn4qa.tensor import Tensor
from tn4qa.tn import TensorNetwork
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import integrals as itg
import psutil
import block2
from pyscf import scf
from scipy.sparse.linalg import eigs 
import sparse
import copy
import numpy as np

class FermionDMRG:

    def __init__(self,
                 scf_obj : scf,
                 HF_symmetry : str,
                 max_mpo_bond : int,
                 max_mps_bond : int,
                 n_core : int=0,
                 n_cas : int=None,
                 g2e_symm : int=1
                 ) -> "FermionDMRG":
        """
        Constructor for the FermionDMRG class. A simple wrapper around Block2 functionality.
        
        Args:
            scf_obj: The (post-HF) scf object.
            HF_symmetry: One of "RHF" or "UHF".
            max_mpo_bond: The maximum bond dimension to use for MPO construction.
            max_mps_bond: The maximum bond dimension to use for MPS during DMRG.
            n_core (optional): The number of core electrons (default 0).
            ncas (optional): The number of electrons in the CAS (default None=all).
            g2e_symm (optional): Symmetry group for 2-electron integrals (default 1).
            
        Returns:
            The FermionDMRG object.
        """
        self.scf_object = scf_obj
        self.HF_symmetry = HF_symmetry
        if self.HF_symmetry == "RHF":
            self.symm_type = SymmetryTypes.SU2
            self.ncas, self.n_elec, self.spin, self.ecore, self.h1e, self.g2e, self.orb_sym = itg.get_rhf_integrals(scf_obj, ncore=n_core, ncas=n_cas, g2e_symm=g2e_symm)
        elif self.HF_symmetry == "UHF":
            self.symm_type = SymmetryTypes.SZ
            self.ncas, self.n_elec, self.spin, self.ecore, self.h1e, self.g2e, self.orb_sym = itg.get_uhf_integrals(scf_obj, ncore=n_core, ncas=n_cas, g2e_symm=g2e_symm)
        else:
            raise ValueError("Unsupported HF symmetry type.")
        self.max_mpo_bond = max_mpo_bond
        self.max_mps_bond = max_mps_bond

        self.driver = self.initialise_driver()

        return
    
    def initialise_driver(self) -> DMRGDriver:
        """
        Initialise the DMRG driver.

        Returns:
            The DMRG driver
        """ 
        n_threads = int(psutil.cpu_count() / psutil.cpu_count(False))
        driver = DMRGDriver(scratch="./tmp", symm_type=self.symm_type, n_threads=n_threads)
        driver.initialize_system(n_sites=self.ncas, n_elec=self.n_elec, spin=self.spin, orb_sym=self.orb_sym)

        return driver
    
    def set_initial_state(self) -> block2:
        """
        Set the initial state for DMRG.
        """
        block2_mps = self.driver.get_random_mps(tag="GS", bond_dim=self.max_mps_bond, nroots=1)

        return block2_mps
    
    def set_hamiltonian(self) -> block2:
        """
        Set the Hamiltonian for DMRG.
        """
        mpo = self.driver.get_qc_mpo(h1e=self.h1e, g2e=self.g2e, ecore=self.ecore, iprint=0)

        return mpo

    def run(self, maxiter : int) -> float:
        """
        Find the groundstate energy of an MPO with DMRG.
        
        Args:
            maxiter: The maximum number of DMRG sweeps.
        
        Returns:
            The DMRG energy.
        """
        mpo = self.set_hamiltonian()
        mps = self.set_initial_state()
        driver = self.driver
        bond_dims = [int(self.max_mps_bond/2)] * 4 + [self.max_mps_bond] * 4
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8
        energy = driver.dmrg(mpo, mps, n_sweeps=maxiter, bond_dims=bond_dims, noises=noises,thrds=thrds, iprint=0)

        return energy
class QubitDMRG:

    def __init__(self, 
                 hamiltonian : dict[str, complex],
                 max_mpo_bond : int,
                 max_mps_bond : int
                 ) -> "QubitDMRG":
        """
        Constructor for the QubitDMRG class.
        
        Args:
            hamiltonian: A dict of the form {pauli_string : weight}.
            max_mpo_bond: The maximum bond to use for the Hamiltonian MPO construction.
            max_mps_bond: The maximum bond to use for MPS during DMRG.
        
        Returns:
            The QubitDMRG object.
        """
        self.hamiltonian = hamiltonian
        self.num_sites = len(list(hamiltonian.keys())[0])
        self.max_mpo_bond = max_mpo_bond
        self.max_mps_bond = max_mps_bond
        self.mps = self.set_initial_state()
        self.mpo = self.set_hamiltonian_mpo()

        return
    
    def set_initial_state(self, mps : MatrixProductState=None) -> MatrixProductState:
        """
        Set the initial state for DMRG.
        
        Args:
            mps (optional): An optional input state. Defaults to random.
        """
        if not mps:
            mps = MatrixProductState.random_quantum_state_mps(self.num_sites, self.max_mps_bond)

        mps = self.add_trivial_tensors_mps(mps)

        return mps
    
    def set_hamiltonian_mpo(self) -> MatrixProductOperator:
        """
        Convert the Hamiltonian to an MPO for DMRG.
        """
        mpo = MatrixProductOperator.from_hamiltonian(self.hamiltonian, self.max_mpo_bond)
        # if mpo.bond_dimension > self.max_mpo_bond:
        #     mpo.compress(self.max_mpo_bond)

        mpo = self.add_trivial_tensors_mpo(mpo)

        return mpo
    
    def add_trivial_tensors_mps(self, mps : MatrixProductState) -> MatrixProductState:
        """
        Add trivial tensors to MPS.
        """ 
        mps.tensors[0].data = sparse.reshape(mps.tensors[0].data, (1,)+mps.tensors[0].dimensions)
        mps.tensors[-1].data = sparse.reshape(mps.tensors[-1].data, (mps.tensors[-1].dimensions[0], 1, mps.tensors[-1].dimensions[1]))

        trivial_array = sparse.COO.from_numpy(np.array([1,1], dtype=complex).reshape(1,2))
        all_arrays = [trivial_array] + [mps.tensors[i].data for i in range(self.num_sites)] + [trivial_array]
        mps = MatrixProductState.from_arrays(all_arrays)
        return mps
    
    def add_trivial_tensors_mpo(self, mpo : MatrixProductOperator) -> MatrixProductOperator:
        """
        Add trivial tensors to MPO.
        """
        mpo.tensors[0].data = sparse.reshape(mpo.tensors[0].data, (1,)+mpo.tensors[0].dimensions)
        mpo.tensors[-1].data = sparse.reshape(mpo.tensors[-1].data, (mpo.tensors[-1].dimensions[0], 1, mpo.tensors[-1].dimensions[1], mpo.tensors[-1].dimensions[2]))
        
        trivial_array = sparse.COO.from_numpy(np.array([[1,1],[1,1]], dtype=complex).reshape(1,2,2))
        all_arrays = [trivial_array] + [mpo.tensors[i].data for i in range(self.num_sites)] + [trivial_array]
        mpo = MatrixProductOperator.from_arrays(all_arrays)
        return mpo
    
    def remove_trivial_tensors_mps(self, mps : MatrixProductState) -> MatrixProductState:
        """
        Remove trivial tensors from MPS.
        """
        first_array = sparse.reshape(mps.tensors[1].data, (mps.tensors[1].dimensions[1], mps.tensors[1].dimensions[2]))
        middle_arrays = [mps.tensors[i].data for i in range(2, self.num_sites-1)]
        last_array = sparse.reshape(mps.tensors[-2].data, (mps.tensors[-2].dimensions[0], mps.tensors[-2].dimensions[2]))
        mps = MatrixProductState.from_arrays([first_array] + middle_arrays + [last_array])
        return mps
    
    def remove_trivial_tensors_mpo(self, mpo : MatrixProductOperator) -> MatrixProductOperator:
        """
        Remove trivial tensors from MPS.
        """ 
        first_array = sparse.reshape(mpo.tensors[1].data, (mpo.tensors[1].dimensions[1], mpo.tensors[1].dimensions[2], mpo.tensors[1].dimensions[3]))
        middle_arrays = [mpo.tensors[i].data for i in range(2, self.num_sites-1)]
        last_array = sparse.reshape(mpo.tensors[-2].data, (mpo.tensors[-2].dimensions[0], mpo.tensors[-2].dimensions[2], mpo.tensors[-2].dimensions[3]))
        mpo = MatrixProductOperator.from_arrays([first_array] + middle_arrays + [last_array])
        return mpo
    
    def construct_expectation_value_tn(self) -> TensorNetwork:
        """
        Construct <mps | mpo | mps>.
        """
        mps = copy.deepcopy(self.mps)
        mps_dag = copy.deepcopy(self.mps)
        mps_dag.dagger()
        mpo = copy.deepcopy(self.mpo)

        num_sites = len(mps.tensors)

        mps.tensors[0].indices = ["b1", "p1"]
        mps_dag.tensors[0].indices = ["bdag1", "pdag1"]
        mpo.tensors[0].indices = ["hb1", "p1", "pdag1"]

        for idx in range(1, num_sites-1):
            mps.tensors[idx].indices = [f"b{idx}", f"b{idx+1}", f"p{idx+1}"]
            mps_dag.tensors[idx].indices = [f"bdag{idx}", f"bdag{idx+1}", f"pdag{idx+1}"]
            mpo.tensors[idx].indices = [f"hb{idx}", f"hb{idx+1}", f"p{idx+1}", f"pdag{idx+1}"]

        mps.tensors[-1].indices = [f"b{num_sites-1}", f"p{num_sites}"]
        mps_dag.tensors[-1].indices = [f"bdag{num_sites-1}", f"pdag{num_sites}"]
        mpo.tensors[-1].indices = [f"hb{num_sites-1}", f"p{num_sites}", f"pdag{num_sites}"]

        all_tensors = mps.tensors + mps_dag.tensors + mpo.tensors 
        tn = TensorNetwork(all_tensors)
        return tn
    
    def get_environment_tensor(self, site_idx : int) -> Tensor:
        """
        Return the environment matrix of the Hamiltonian at given site.
        
        Args:
            site_idx: The site index.
        
        Returns:
            A Tensor object with indices [udag, pdag, ddag, u, p, d]
        """
        tn = self.construct_expectation_value_tn()
        tn.pop_tensors_by_label([f"MPS_T{site_idx}"])
        env_tensor = tn.contract_entire_network()
        env_tensor.reorder_indices([f"bdag{site_idx-1}", f"pdag{site_idx}", f"bdag{site_idx}", f"b{site_idx-1}", f"p{site_idx}", f"b{site_idx}"])
        env_tensor.indices = ["udag", "pdag", "ddag", "u", "p", "d"]

        return env_tensor
    
    def sweep(self, direction : str) -> float:
        """
        Perform one DMRG sweep.
        
        Args:
            direction: Either "F" (forward) or "B" (backward)

        Return:
            The current groundstate energy estimate.
        """
        sites = list(range(2, self.num_sites+2))
        if direction == "B": sites = sites[::-1]

        for site in sites:
            self.mps.move_orthogonality_centre(site)
            original_dims = self.mps.tensors[site-1].dimensions
            env_tensor = self.get_environment_tensor(site)
            env_tensor.tensor_to_matrix(["u", "p", "d"], ["udag", "pdag", "ddag"])
            w, v = eigs(env_tensor.data, k=1, which="SR")
            eigval = w[0]
            eigvec = sparse.COO.from_numpy(v[:, 0]) # This is the new optimal value at site i
            new_data = sparse.reshape(eigvec, (original_dims[0], original_dims[2], original_dims[1]))
            new_data = sparse.moveaxis(new_data, [0,1,2], [0,2,1])

            original_indices = self.mps.tensors[site-1].indices
            original_labels = self.mps.tensors[site-1].labels
            self.mps.pop_tensors_by_label(original_labels)
            new_t = Tensor(new_data, original_indices, original_labels)
            self.mps.add_tensor(new_t, site-1)
        
        return eigval

    def run(self, maxiter : int) -> Tuple[float, MatrixProductState]:
        """
        Find the groundstate of an MPO with DMRG.
        
        Args:
            max_mps_bond: The maximum bond dimension allowed. 
            maxiter: The maximum number of DMRG sweeps.
        
        Returns:
            A tuple of the DMRG energy and the DMRG state.
        """
        for _ in range(maxiter):
            e = self.sweep("F")
            e = self.sweep("B")
        
        energy = e.real / 4 # Trivial end tensors mean we overcount the energy by factor of 4

        return (energy, self.mps)
    