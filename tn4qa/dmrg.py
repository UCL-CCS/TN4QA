from typing import Tuple 
from .mps import MatrixProductState
from .mpo import MatrixProductOperator
import block2 

class DMRG:

    def __init__(self, 
                 hamiltonian : dict[str, complex],
                 max_mpo_bond : int
                 ) -> "DMRG":
          self.hamiltonian = hamiltonian
          self.max_mpo_bond = max_mpo_bond
          self.mpo = MatrixProductOperator.from_hamiltonian(hamiltonian, max_mpo_bond)
        
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
