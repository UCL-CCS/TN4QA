from typing import Tuple 
from .mps import MatrixProductState
from .mpo import MatrixProductOperator
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import integrals as itg
import psutil
import block2
from pyscf import scf


class FermionDMRG:

    def __init__(self,
                 scf_obj : scf,
                 HF_symmetry : str,
                 max_mpo_bond : int,
                 max_mps_bond : int,
                 n_core : int=0,
                 n_cas : int=None,
                 g2e_symm : int=8
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
            g2e_symm (optional): Symmetry group for 2-electron integrals (default 8).
            
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
        self.max_mpo_bond = max_mpo_bond
        self.max_mps_bond = max_mps_bond
        self.num_orbitals = None
        self.num_electrons = None
        self.spin = None

        self.tn4qa_mpo = MatrixProductOperator.from_hamiltonian(hamiltonian, max_mpo_bond)
        self.block2_mpo = self.tn4qa_mpo.to_block2_mpo()
        self.driver = self.initialise_driver()

        return
    
    def initialise_driver(self) -> DMRGDriver:
        """
        Initialise the DMRG driver.

        Returns:
            The DMRG driver
        """
        if self.HF_symmetry == "RHF":
            symm_type = SymmetryTypes.SU2
        else:
            symm_type = SymmetryTypes.SZ
        
        n_threads = psutil.cpu_count()
        driver = DMRGDriver(scratch="./tmp", symm_type=symm_type, n_threads=n_threads)
        driver.initialize_system(n_sites=self.num_orbitals, n_elec=self.num_electrons, spin=self.spin, orb_sym=symm_type)

        return driver
    
    def set_initial_state(self, max_mps_bond : int, mps : MatrixProductState=None) -> block2:
        """
        Set the initial state for DMRG.
        
        Args:
            max_mps_bond: The maximum bond dimension allows for the MPS. 
            mps (optional): An optional input state. Defaults to random.
        """
        if not mps:
            block2_mps = self.driver.get_random_mps(tag="GS", bond_dim=max_mps_bond, nroots=1)
        else:
            block2_mps = mps.to_block2_mps()
        return block2_mps


    def run(self, max_mps_bond : int, maxiter : int) -> Tuple[float, MatrixProductState]:
        """
        Find the groundstate of an MPO with DMRG.
        
        Args:
            max_mps_bond: The maximum bond dimension allowed. 
            maxiter: The maximum number of DMRG sweeps.
        
        Returns:
            A tuple of the DMRG energy and the DMRG state.
        """
        mpo = self.block2_mpo 
        mps = self.set_initial_state(max_mps_bond=max_mps_bond)
        driver = self.driver
        bond_dims = [int(max_mps_bond/2)] * 4 + [max_mps_bond] * 4
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8
        energy = driver.dmrg(mpo, mps, n_sweeps=maxiter, bond_dims=bond_dims, noises=noises,thrds=thrds, iprint=0)
        tn4qa_mps = MatrixProductState.from_block2_mps(mps)

        return (energy, tn4qa_mps)
