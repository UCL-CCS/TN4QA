from typing import Tuple 
from .mps import MatrixProductState
from .mpo import MatrixProductOperator
from pyscf import scf
import block2
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
import psutil

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

        self.tn4qa_mpo = MatrixProductOperator.from_hamiltonian(hamiltonian, max_mpo_bond)
        self.block2_mpo = self.tn4qa_mpo.to_block2_mpo()

        return

    @classmethod
    def from_scf(cls, scf_obj : scf, max_mpo_bond : int, qubit_transformation : str="JW") -> "DMRG":
        """
        Construct the DMRG class from an scf object.
        
        Args:
            scf_obj: The scf object (post-HF)
            max_mpo_bond: The maximum bond dimension to construct the MPO
            qubit_transformation (optional): The fermion-to-qubit mapping to use (default JW)
            
        Returns:
            The class instance.
        """
        if scf_obj.istype("RHF"):
            HF_symmetry = "RHF"
        elif scf_obj.istype("UHF"):
            HF_symmetry = "UHF"
        else:
            raise TypeError("Unsupported HF symmetry type.")
        
        ham = None

        return cls(ham, HF_symmetry, max_mpo_bond)
    
    def get_block2_mpo_from_scf(self, scf_obj : scf) -> block2:
        """
        Generate the Block2 MPO object from the scf object.
        
        Args:
            scf_obj: The PySCF scf object (post-HF).
        
        Returns:
            The Block2 MPO.
        """
        if self.HF_symmetry == "RHF":
            symm_type = SymmetryTypes.SU2
        else:
            symm_type = SymmetryTypes.SZ
        
        n_threads = psutil.cpu_count()
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(scf_obj, ncore=0, ncas=None, g2e_symm=8)
        driver = DMRGDriver(scratch="./tmp", symm_type=symm_type, n_threads=n_threads)
        driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)

        mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
        return mpo
    


    def run(self, max_bond : int, maxiter : int) -> Tuple[float, MatrixProductState]:
        """
        Find the groundstate of an MPO with DMRG.
        
        Args:
            max_bond: The maximum bond dimension allowed. 
            maxiter: The maximum number of DMRG sweeps.
        
        Returns:
            A tuple of the DMRG energy and the DMRG state.
        """
        return
