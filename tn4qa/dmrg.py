from typing import Tuple 
from .mps import MatrixProductState
from .mpo import MatrixProductOperator
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
import psutil
import block2

class DMRG:

    def __init__(self, 
                 hamiltonian : dict[str, complex],
                 HF_symmetry : str,
                 max_mpo_bond : int
                 ) -> "DMRG":
        """
        Constructor for the DMRG class.
        
        Args:
            hamiltonian: A dict of the form {pauli_string : weight}.
            HF_symmetry: The symmetry type used for HF calculation, either RHF or UHF. 
            max_mpo_bond: The maximum bond to use for the Hamiltonian MPO construction.
        
        Returns:
            The DMRG object.
        """
        self.hamiltonian = hamiltonian
        self.HF_symmetry = HF_symmetry
        self.max_mpo_bond = max_mpo_bond
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
