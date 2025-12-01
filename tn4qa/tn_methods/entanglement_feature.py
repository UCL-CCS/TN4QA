import numpy as np
from tn4qa.mps import MatrixProductState as MPS

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
                    # swap: |c>⊗|c> goes to |c>⊗|c>
                    T1[a, a, c, b, b, c] = 1.0

        T_site = np.zeros((2, u, u, p, d, d, p))
        T_site[0] = T0
        T_site[1] = T1

        EF.append(T_site)

    return EF

# Contract EF with a given bitstring
def contract_ef_bitstring(mps, EF, bitstring):
    """
    Compute <psi⊗psi | EF(bitstring) | psi⊗psi>
    Contract the entanglement feature EF with a given bitstring.
    bitstring: list of 0/1 of length N
    Returns: Renyi-2 value 
    """
    # extract MPS tensors
    A_list = [getattr(A, "data", A) for A in mps.tensors]
    N = len(A_list)

    # left boundary environment: scalar 1
    env = np.array([1.0]).reshape(1, 1)

    for i in range(N):
        A = A_list[i]                 # (u, d, p)
        T = EF[i][bitstring[i]]  # (u, u, p, d, d, p)

        # A ⊗ A
        A2 = np.tensordot(A, A, axes=0)  # (u1, d1, p1, u2, d2, p2)

        # Contract right physical leggies
        # A2* with T
        X = np.tensordot(A2.conj(), T , axes=([2, 3], [1, 2]))  # (u, d, u, d, p, p) 

        # Contract the other physical leggies
        # X with A2
        X = np.tensordot(X, A2, axes=([4,5,6,7], [0,3,1,4]))  # (u1,d1,u2,d2,  u1'',d1'',u2'',d2'')

        # Merge legs to form a matrix for environment update
        # left
        left = (X.shape[0], X.shape[1])*(X.shape[2], X.shape[3])
        # right
        right = (X.shape[4], X.shape[5])*(X.shape[6], X.shape[7])
        Xmat = X.reshape(left, right)

        # Update environment
        env = np.dot(env, Xmat)  # (new left dim, new right dim)

    # Final contraction to get scalar
    return env.squeeze()


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
        R2 = contract_ef_bitstring(mps, EF, bitstring)
        costs.append(R2)
        best_cut = np.argmin(costs) + 1
    return best_cut


def split_mps_at_cut(mps, cut):
    left_tensors = mps.tensors[:cut]
    right_tensors = mps.tensors[cut:]

    left = MPS(left_tensors)
    right = MPS(right_tensors)
    return left, right