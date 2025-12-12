import numpy as np
from tn4qa.mps import MatrixProductState as MPS
from tn4qa.mpo import MatrixProductOperator as MPO
from tn4qa.tn import TensorNetwork as TN

def build_entanglement_feature(mps):
    """
    Build entanglement feature (EF) for MPS.
    Returns EF as a list of tensors, one per site
    Each tensor has shape (2, dim_up, dim_up, phys_dim, dim_down, dim_down, phys_dim)
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

        # Identity (bit=0)
        T0 = np.zeros((u, u, p, d, d, p))
        # Swap up/down (bit=1)
        T1 = np.zeros((u, u, p, d, d, p))
        for a in range(u):
            for b in range(d):
                for c in range(p):
                    # identity: |c>⊗|c> stays |c>⊗|c>
                    T0[a, a, c, b, b, c] = 1.0
                for c_prime in range(p):
                    for c in range(p):
                        # swap: |c>⊗|c'> goes to |c'>⊗|c>
                        T1[a, a, c_prime, b, b, c] = 1.0

        T_site = np.zeros((2, u, u, p, d, d, p))
        T_site[0] = T0
        T_site[1] = T1

        EF.append(T_site)

    return EF

# Contract EF with a given bitstring
def contract_ef_bitstring(EF, bitstring):
    """
    Compute <psi⊗psi | EF(bitstring) | psi⊗psi>
    Contract the entanglement feature EF with a given bitstring.
    bitstring: list of 0/1 of length N
    Returns: Renyi-2 value (float)
    """
    assert len(bitstring) == len(EF)
    N = len(EF)

    # Use trace(A ⊗ B) = trace(A) * trace(B) to avoid exponential growth of matrix size
    R2 = 1.0    # would be great if true but i dont think it is lol
    
    for i in range(N):
        Ti = EF[i][bitstring[i]]        # shape (u, u, p, d, d, p)
        left = np.prod(Ti.shape[:3])   # dimensions to the left of the trace
        right = np.prod(Ti.shape[3:])   # dimensions to the right of the trace
        Ti_reshaped = Ti.reshape((left, right)) # shape (left, right)
        print("Ti_reshaped shape:", Ti_reshaped.shape)
        R2 *= np.trace(Ti_reshaped)   # trace over the reshaped matrix

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