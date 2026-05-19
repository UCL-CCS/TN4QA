import copy

import numpy as np

from tn4qa.mps import MatrixProductState as MPS
from tn4qa.tensor import Tensor
from tn4qa.tn import TensorNetwork as TN


def build_entanglement_feature(mps: MPS) -> MPS:
    """
    Build entanglement feature (EF) for MPS.
    Returns EF as an MPS
    """
    ef_arrays = []
    x = mps.bond_dimension
    for tidx in range(1, mps.num_sites):
        t = mps.tensors[tidx]
        bond_size = t.data.shape[0]
        if bond_size < x:
            mps = mps.expand_bond_dimension(x-bond_size, tidx)
    for A in mps.tensors:
        if A.data.ndim == 2:
            x, p = A.data.shape  # A shape (x,2)
            print("Bond dim:", x, "Physical dim:", p)
            assert p == 2, "Physical dimension must be 2"
            A1 = copy.deepcopy(A)  # shape (x,2)
            A2 = copy.deepcopy(A)  # shape (x,2)
            # Set indices
            A1.indices = ["u1", "p1"]
            A2.indices = ["u2", "p2"]
            # Build D tensor = A1 * A2 of shape (x^2, 2, 2)
            D = np.zeros((x**2, 2, 2))
            D[:, 0, 0] = np.kron(A1.data[:, 0], A2.data[:, 0])
            D[:, 1, 1] = np.kron(A1.data[:, 1], A2.data[:, 1])
            D[:, 0, 1] = np.kron(A1.data[:, 0], A2.data[:, 1])
            D[:, 1, 0] = np.kron(A1.data[:, 1], A2.data[:, 0])
            D = Tensor(D, indices=["a", "c", "d"], labels=["Double_Tensor"])
            print("Shape of D tensor:", D.data.shape)
            print("Labels of D:", D.labels)
            print("Indices of D:", D.indices)

            D_conj = copy.deepcopy(D)
            D_conj.dagger()
            # Make two tensor networks: one for identity and one for swap
            D_conj.indices = ["e", "c", "d"]
            tn_id = TN([D, D_conj])  # identity
            print("TN identity tensors:", [t.indices for t in tn_id.tensors])

            D_conj.indices = ["e", "d", "c"]
            tn_sw = TN([D, D_conj])  # swap

            # Contract c and d, combine a,e and b,f
            # to get final tensor network of one tensor of shape (x^4, x^4)
            tn_id.contract_indices(["c", "d"])
            print(
                "After contracting, TN identity tensors:",
                [t.indices for t in tn_id.tensors],
            )
            print(
                "Before combining, TN identity tensors:",
                [t.indices for t in tn_id.tensors],
            )
            tn_id.tensors[0].combine_indices(["a", "e"], "up")
            print(
                "After combining, TN identity tensors:",
                [t.indices for t in tn_id.tensors],
            )
            print("TN Identity tensor shape:", tn_id.tensors[0].data.shape)

            tn_sw.contract_indices(["c", "d"])
            tn_sw.tensors[0].combine_indices(["a", "e"], "up")
            print("TN Swap tensor shape:", tn_sw.tensors[0].data.shape)

            # Make an array with the tensors
            T = np.zeros((x**4, 2))

            ef_tensor_id = tn_id.tensors[
                0
            ]  # extract the single tensor from the identity TN
            T[:, 0] = ef_tensor_id.to_dense()  # add to array
            ef_tensor_sw = tn_sw.tensors[
                0
            ]  # extract the single tensor from the swap TN
            T[:, 1] = ef_tensor_sw.to_dense()  # add to array

            ef_arrays.append(T)

        elif A.data.ndim == 3:
            x, x, p = A.data.shape  # A shape (x,x,2)
            print("Bond dim:", x, "Physical dim:", p)
            assert p == 2, "Physical dimension must be 2"
            A1 = copy.deepcopy(A)  # shape (x,x,2)
            A2 = copy.deepcopy(A)  # shape (x,x,2)

            # Set indices
            A1.indices = ["u1", "d1", "p1"]
            A2.indices = ["u2", "d2", "p2"]

            # Build D tensor = A1 * A2 of shape (x^2, x^2, 2, 2)
            D = np.zeros((x**2, x**2, 2, 2))
            D[:, :, 0, 0] = np.kron(A1.data[:, :, 0], A2.data[:, :, 0])
            D[:, :, 1, 1] = np.kron(A1.data[:, :, 1], A2.data[:, :, 1])
            D[:, :, 0, 1] = np.kron(A1.data[:, :, 0], A2.data[:, :, 1])
            D[:, :, 1, 0] = np.kron(A1.data[:, :, 1], A2.data[:, :, 0])
            D = Tensor(D, indices=["a", "b", "c", "d"], labels=["Double_Tensor"])
            print("Shape of D tensor:", D.data.shape)
            print("Labels of D:", D.labels)
            print("Indices of D:", D.indices)

            D_conj = copy.deepcopy(D)
            D_conj.dagger()

            # Make two tensor networks: one for identity and one for swap
            D_conj.indices = ["e", "f", "c", "d"]
            tn_id = TN([D, D_conj])  # identity
            print("TN identity tensors:", [t.indices for t in tn_id.tensors])

            D_conj.indices = ["e", "f", "d", "c"]
            tn_sw = TN([D, D_conj])  # swap

            # Contract c and d, combine a,e and b,f
            # to get final tensor network of one tensor of shape (x^4, x^4)
            tn_id.contract_indices(["c", "d"])
            print(
                "After contracting, TN identity tensors:",
                [t.indices for t in tn_id.tensors],
            )
            print(
                "Before combining, TN identity tensors:",
                [t.indices for t in tn_id.tensors],
            )
            tn_id.tensors[0].combine_indices(["a", "e"], "up")
            tn_id.tensors[0].combine_indices(["b", "f"], "down")
            print(
                "After combining, TN identity tensors:",
                [t.indices for t in tn_id.tensors],
            )
            print("TN Identity tensor shape:", tn_id.tensors[0].data.shape)

            tn_sw.contract_indices(["c", "d"])
            tn_sw.tensors[0].combine_indices(["a", "e"], "up")
            tn_sw.tensors[0].combine_indices(["b", "f"], "down")
            print("TN Swap tensor shape:", tn_sw.tensors[0].data.shape)

            # Make an array with the tensors
            T = np.zeros((x**4, x**4, 2))

            ef_tensor_id = tn_id.tensors[
                0
            ]  # extract the single tensor from the identity TN
            T[:, :, 0] = ef_tensor_id.to_dense()  # add to array
            ef_tensor_sw = tn_sw.tensors[
                0
            ]  # extract the single tensor from the swap TN
            T[:, :, 1] = ef_tensor_sw.to_dense()  # add to array

            ef_arrays.append(T)
    EF = MPS.from_arrays(ef_arrays)  # build MPS from array of tensors
    return EF


# Contract EF with a given bitstring
def contract_ef_bitstring(ef_mps: MPS, bitstring: list[int]) -> float:
    """
    Compute <psi⊗psi | EF(bitstring) | psi⊗psi>
    Contract the entanglement feature EF with a given bitstring.
    bitstring: list of 0/1 of length N
    Returns: Renyi-2 value (float)
    """
    bit_mps = MPS.from_bitstring(bitstring)
    a = ef_mps.compute_inner_product(bit_mps)
    R2 = a.real
    return R2


# best cut according to EF
def ef_best_cut(ef_mps: MPS) -> int:
    """
    Return the cut index i that minimises the Renyi-2 across bitstrings
    """
    N = len(ef_mps.tensors)
    print("Calculating best cut for MPS with", N, "tensors")
    costs = []
    for i in range(1, N):
        bitstring = [1] * i + [0] * (N - i)
        R2 = contract_ef_bitstring(ef_mps, bitstring)
        costs.append(R2)
        best_cut = np.argmin(costs) + 1
    return best_cut


def split_mps_at_cut(mps, cut):
    left_tensors = mps.tensors[:cut]
    right_tensors = mps.tensors[cut:]

    left = MPS(left_tensors)
    right = MPS(right_tensors)
    return left, right
