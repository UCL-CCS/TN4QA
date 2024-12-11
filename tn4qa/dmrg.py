from typing import Tuple

from .mpo import MatrixProductOperator
from .mps import MatrixProductState


class DMRG:
    def run(
        self, mpo: MatrixProductOperator, max_bond: int, maxiter: int
    ) -> Tuple[float, MatrixProductState]:
        """
        Find the groundstate of an MPO with DMRG.

        Args:
            max_bond: The maximum bond dimension allowed.
            maxiter: The maximum number of DMRG sweeps.

        Returns:
            A tuple of the DMRG energy and the DMRG state.
        """
        return
