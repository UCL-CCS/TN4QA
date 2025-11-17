import numpy as np

def build_entanglement_feature(mps):
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
