import numpy as np
from tn4qa.mps import MatrixProductState as MPS
from tn4qa.mpo import MatrixProductOperator as MPO
from tn4qa.tn import TensorNetwork as TN
from tn4qa.tensor import Tensor

def build_entanglement_feature(mps):
    """
    Build entanglement feature (EF) for MPS.
    Returns EF as an MPS
    """
    ef_tensors = []

    for A in mps.tensors:
        A1 = A.copy()
        A2 = A.copy()

        # Set index labels
        A1.set_index_labels(['u1', 'd1', 'p1'])
        A2.set_index_labels(['u2', 'd2', 'p2'])

        D = A1 @ A2
        D = D.combine_indices([('u1', 'u2')], new_label='a')
        D = D.combine_indices([('d1', 'd2')], new_label='b')
        D.set_index_labels(['a', 'b', 'c', 'd'])

        D_conj = D.dagger()
        D_conj.set_index_labels(['e', 'f', 'p1', 'p2'])

        for p1 in [0,1]:
            for p2 in [0,1]:
                m1 = A1[:,:p1]
                m2 = A2[:,:p2]
                Terry[:,:,p1,p2] = m1 @ m2

        if T[:,:,0] :
            D_conj.set_index_labels(['e', 'f', 'c', 'd'])
            Terry = TN(D, D_conj)

        if SWAP:
            D_conj.set_index_labels(['e', 'f', 'd', 'c'])
            Terry = TN(D, D_conj)
            Terry.contract_indexs(['c', 'd'])
            Terry.combine_indices([('a', 'e')], new_label='up')
            Terry.combine_indices([('b', 'f')], new_label='down') 


    return EF

# Contract EF with a given bitstring
def contract_ef_bitstring(EF, bitstring):
    """
    Compute <psi⊗psi | EF(bitstring) | psi⊗psi>
    Contract the entanglement feature EF with a given bitstring.
    bitstring: list of 0/1 of length N
    Returns: Renyi-2 value (float)
    """
    assert len(bitstring) == len(EF) , "Bitstring length must match EF length"
    ef_mps = MPS.from_arrays(EF)
    bit_mps = MPS.from_bitstring(bitstring)
    a = ef_mps.compute_inner_product(bit_mps)
    R2 = a.real
    return R2

# best cut according to EF
def ef_best_cut(mps):
    """
    Return the cut index i that minimises the Renyi-2 across bitstrings
    """
    N = len(mps.tensors)
    costs = []
    EF = build_entanglement_feature(mps)
    for i in range(1,N):
        bitstring = [1]*i + [0]*(N-i)
        R2 = contract_ef_bitstring(EF, bitstring)
        costs.append(R2)
        best_cut = np.argmin(costs) + 1
    return best_cut


def split_mps_at_cut(mps, cut):
    left_tensors = mps.tensors[:cut]
    right_tensors = mps.tensors[cut:]

    left = MPS(left_tensors)
    right = MPS(right_tensors)
    return left, right