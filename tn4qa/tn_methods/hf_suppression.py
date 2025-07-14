import copy

import numpy as np

from ..mpo import MatrixProductOperator
from ..mps import MatrixProductState


class HFSuppression:
    def __init__(
        self,
        hf_state: str,
        mps: MatrixProductState,
        mpo: MatrixProductOperator | None = None,
    ):
        """Constructor

        Args:
            hf_state: The HF state of the system
            mps: The state to suppress the HF contribution
            mpo: The operator to evolve
        """
        self.hf_state = hf_state
        self.num_electrons = hf_state.count("1")
        self.num_qubits = mps.num_sites
        self.hf_state_mps = MatrixProductState.from_bitstring(hf_state)
        self.mps = mps
        self.mpo = mpo

    def build_hf_oracle(self) -> MatrixProductOperator:
        """Build oracle for HF amplitude suppression"""
        oracle_mpo = self.hf_state_mps.form_density_operator()
        oracle_mpo.multiply_by_constant(2.0)
        id_mpo = MatrixProductOperator.identity_mpo(oracle_mpo.num_sites)
        oracle_mpo = oracle_mpo - id_mpo

        return oracle_mpo

    def build_diffusion_operator(self) -> MatrixProductOperator:
        """Build diffusion operator circuit"""
        diffusion_mpo = self.mps.form_density_operator()
        diffusion_mpo.multiply_by_constant(2.0)
        id_mpo = MatrixProductOperator.identity_mpo(self.mps.num_sites)
        diffusion_mpo = diffusion_mpo - id_mpo

        return diffusion_mpo

    def hf_suppression(self, max_bond: int | None = None) -> MatrixProductState:
        """Perform HF suppression on MPS

        Args:
            max_bond: Maximum bond dimension

        Returns:
            The HF suppressed MPS
        """
        oracle = self.build_hf_oracle()
        diffusion = self.build_diffusion_operator()
        hf_amplitude = self.mps.compute_inner_product(self.hf_state_mps)
        rotation_angle = np.sqrt(1 - np.abs(hf_amplitude) ** 2)
        num_iterations = int(np.floor(np.pi / (4 * rotation_angle)))
        current_mps = copy.deepcopy(self.mps)
        for idx in range(num_iterations):
            current_mps = current_mps.apply_mpo(oracle, max_bond)
            current_mps = current_mps.apply_mpo(diffusion, max_bond)
        return current_mps

    def run(self):
        """Run HF Suppression

        Args:
            method: Either "subtract" or "amplitude_amplification" """
        return
