import numpy as np
from tn4qa.mps import MatrixProductState as MPS

def build_entanglement_feature(mps):
    """
    Build entanglement feature (EF) for MPS.
    """
    EF = []

    for A in mps.tensors:
        data = getattr(A, "data", A)
        shape = data.shape

        if len(shape) == 2:
            # Boundary tensor: (up, phys)
            u, p = shape
            d = 1
        elif len(shape) == 3:
            # Bulk tensor: (up, down, phys)
            u, d, p = shape
        else:
            raise ValueError(f"Unexpected tensor shape {shape}")

        dim = u * d * p

        # Identity (bit 0)
        T0 = np.eye(dim)

        # Swap up/down (bit=1)
        T1 = np.zeros((dim, dim))
        for a in range(u):
            for b in range(d):
                for c in range(p):
                    old = a * d * p + b * p + c
                    new = b * u * p + a * p + c
                    T1[old, new] = 1.0

        T_site = np.zeros((2, dim, dim))
        T_site[0] = T0
        T_site[1] = T1

        EF.append(T_site)

    return EF

# Contract EF with a given bitstring
def contract_ef_bitstring(EF, bitstring):
    """
    Contract the entanglement feature EF with a given bitstring.
    bitstring: list of 0/1 of length N
    Returns: Renyi-2 value (float)
    """
    assert len(bitstring) == len(EF)
    N = len(EF)

    # Start with the first site
    C = EF[0][bitstring[0]]

    for i in range(1, N):
        C = np.kron(C, EF[i][bitstring[i]])

    # Trace to get Renyi-2
    R2 = np.trace(C)
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
