import numpy as np
from mps import MatrixProductState
from qi_metrics import get_one_orbital_entropy, get_all_mutual_information

def cost_entropy(mps: MatrixProductState) -> float:
    """
    Cost based on total single-orbital entropy.
    """
    num_orbitals = mps.num_sites // 2
    return sum(get_one_orbital_entropy(mps, i + 1) for i in range(num_orbitals))

def cost_total_mutual_information(mps: MatrixProductState) -> float:
    """
    Sum of all mutual informations. Lower total means less entanglement.
    """
    mi = get_all_mutual_information(mps)
    return np.sum(mi) / 2  # MI is symmetric

def cost_mutual_info_decay(mps: MatrixProductState, decay_power: float = 2.0) -> float:
    """
    Cost function penalising long-range mutual information.
    Put highly entangled orbitals next to each other in the DMRG chain
    """
    mi = get_all_mutual_information(mps)
    n_orbs = mi.shape[0]
    cost = 0.0
    for i in range(n_orbs):
        for j in range(i + 1, n_orbs):
            distance = abs(i - j)
            cost += mi[i, j] * (distance ** decay_power)
    return cost

def cost_mutual_info_clusters(mps: MatrixProductState, threshold: float = 0.1) -> float:
    """
    Cost is the number of orbital pairs with mutual information above threshold that are far apart.
    """
    mi = get_all_mutual_information(mps)
    n_orbs = mi.shape[0]
    cost = 0.0
    for i in range(n_orbs):
        for j in range(i + 1, n_orbs):
            if mi[i, j] > threshold:
                cost += abs(i - j)
    return cost

def cost_crossing_mi_pairs(mps: MatrixProductState, threshold: float = 0.1) -> float:
    """
    Cost is the number of pairs of orbitals with mutual information above threshold that cross.
    Avoid high-MI pairs "crossing over" each other in the ordering
    """
    mi = get_all_mutual_information(mps)
    n_orbs = mi.shape[0]
    crossings = 0
    for i in range(n_orbs):
        for j in range(i + 1, n_orbs):
            if mi[i, j] < threshold:
                continue
            for k in range(i + 1, j):
                for l in range(j + 1, n_orbs):
                    if mi[k, l] > threshold and (i < k < j < l or k < i < l < j):
                        crossings += 1
    return crossings

def cost_entropy_max_to_mean(mps: MatrixProductState) -> float:
    """
    Cost based on the ratio of the maximum single-orbital entropy to the mean single-orbital entropy.
    Encourage a sharp entropy distribution
    → a few orbitals with high entanglement (to be kept in the active space)
    → and many with low entropy (to be discarded or treated classically)
    """
    entropies = [get_one_orbital_entropy(mps, i + 1) for i in range(mps.num_sites // 2)]
    mean = np.mean(entropies)
    return max(entropies) / mean if mean != 0 else -np.inf
