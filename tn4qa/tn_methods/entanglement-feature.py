import numpy as np

def build_entanglement_feature(mps):
    """
    Build the entanglement feature (EF) for a given MPS.
    """
    N = len(mps.tensors)
    EF = []
    for i in range(N):
        A = mps[i]
        l, r = A.shape[0], A.shape[2]
        # identity  
        T0 = np.zeros((l*r, l*r))
        for a in range(l):
            for b in range(r):
                T0[a*r + b, a*r + b] = 1.0
        # swap
        T1 = np.zeros((l*r, l*r))
        for a in range(l):
            for b in range(r):
                T1[a*r+b, b*l+a] = 1.0

        T_site = np.zeros((2, l*r, l*r))
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
    # most left first 
    C = EF[0][bitstring[0]]     # (χ0 × χ0)
    # contract from left to right
    for i in range(1,N):
        C = C @ EF[i][bitstring[i]]   # (χi × χi)
    R2 = np.trace(C)
    return R2

# best cut according to EF
def ef_best_cut(mps):
    """
    Return the cut index i that minimises the Renyi-2 across bitstrings:
    left=1...1 (i times), right=0...0.
    """
    N = len(mps.tensors)
    costs = []
    for i in range(1,N):
        bitstring = [1]*i + [0]*(N-i)
        R2 = contract_ef_bitstring(bitstring)
        costs.append(R2)
        best_cut = np.argmin(costs) + 1
    return best_cut


def split_mps_at_cut(mps, cut):
    left = mps.subchain(0, cut)
    right = mps.subchain(cut, mps.num_sites)
    return left, right