import numpy as np
from qiskit import QuantumCircuit

from ..mpo import MatrixProductOperator
from ..mps import MatrixProductState
from .mps_to_circuit import MPSAnalyticDecomposition


class LogDepthVerifierCircuit:
    def __init__(self, mpo: MatrixProductOperator):
        self.mpo = mpo

    def build(self, max_svd_dim: int | None = None) -> QuantumCircuit:
        self.mps = MatrixProductState.from_mpo(self.mpo)
        self.norm = self.mps.compute_inner_product(self.mps).real
        self.mps.multiply_by_constant(np.sqrt(1 / self.norm))
        self.mps = pad_bond_dim(self.mps)
        decomp = MPSAnalyticDecomposition(self.mps, 1, 1.0)
        qc = decomp.mps_to_qc_via_ttn(self.mps, max_svd_dim)
        verifier_circ = qc.inverse()
        return verifier_circ


def pad_bond_dim(mps: MatrixProductState):
    bond_dim = mps.bond_dimension
    padded_bond_dim = 1
    while padded_bond_dim < bond_dim:
        padded_bond_dim = 2 * padded_bond_dim

    for idx in range(1, mps.num_sites):
        bond_dim = mps.tensors[idx].dimensions[0]
        if bond_dim < padded_bond_dim:
            mps = mps.expand_bond_dimension(padded_bond_dim - bond_dim, idx)
    return mps
