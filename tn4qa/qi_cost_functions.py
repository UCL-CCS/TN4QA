from typing import Callable

import numpy as np
from numpy import ndarray

from .mpo import MatrixProductOperator
from .mps import MatrixProductState
from .qi_metrics import (
    get_all_mutual_information,
    get_one_orbital_entropy,
    get_one_orbital_rdm,
    get_two_orbital_rdm,
)


def cost_entropy(mps: MatrixProductState) -> float:
    """
    Cost based on total single-orbital entropy.
    """
    num_orbitals = mps.num_sites // 2
    return sum(get_one_orbital_entropy(mps, i + 1) for i in range(num_orbitals))


def cost_entropy_dict(num_orbitals: int) -> dict:
    s = {}
    for i in range(num_orbitals):
        s[f"S1_{i+1}"] = 1.0
    return s


def cost_total_mutual_information(mps: MatrixProductState) -> float:
    """
    Sum of all mutual informations. Lower total means less entanglement.
    """
    mi = get_all_mutual_information(mps)
    return np.sum(mi) / 2  # MI is symmetric


def cost_total_mutual_information_dict(num_orbitals: int) -> dict:
    s = {}
    for i in range(num_orbitals):
        s[f"S1_{i+1}"] = s.get(f"S1_{i+1}", 0) + 1.0
        for j in range(i + 1, num_orbitals):
            s[f"S1_{i+1}"] = s.get(f"S1_{i+1}", 0) + 1.0
            s[f"S1_{j+1}"] = s.get(f"S1_{j+1}", 0) + 1.0
            s[f"S2_{i+1}_{j+1}"] = s.get(f"S2_{i+1}_{j+1}", 0) - 1.0
    return s


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
            cost += mi[i, j] * (distance**decay_power)
    return cost


def cost_mutual_info_decay_dict(num_orbitals: int, decay_power: float = 2.0) -> dict:
    s = {}
    for i in range(num_orbitals):
        for j in range(i + 1, num_orbitals):
            distance = abs(i - j)
            s[f"S1_{i+1}"] = s.get(f"S1_{i+1}", 0) + (distance**decay_power)
            s[f"S1_{j+1}"] = s.get(f"S1_{j+1}", 0) + (distance**decay_power)
            s[f"S2_{i+1}_{j+1}"] = s.get(f"S2_{i+1}_{j+1}", 0) - (distance**decay_power)
    return s


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


def cost_function_to_dict(cost_function: Callable, **kwargs) -> dict[str, float]:
    function_params = kwargs
    match cost_function.__name__:
        case "cost_entropy":
            num_orbitals = function_params["num_orbitals"]
            return cost_entropy_dict(num_orbitals=num_orbitals)
        case "cost_total_mutual_information":
            num_orbitals = function_params["num_orbitals"]
            return cost_total_mutual_information_dict(num_orbitals=num_orbitals)
        case "cost_mutual_info_decay":
            num_orbitals = function_params["num_orbitals"]
            decay_power = function_params["decay_power"]
            return cost_mutual_info_decay_dict(num_orbitals=num_orbitals, decay_power=decay_power)
        case _:
            raise ValueError
    return


def calculate_purity(density_matrix: ndarray) -> float:
    rho_squared = density_matrix @ density_matrix
    purity = np.trace(rho_squared)
    return purity


def cost_function_dict_to_callable(
    cost_function_dict: dict[str, float], entropy_function: Callable[[ndarray], float]
) -> Callable[[MatrixProductState], float]:
    def cost_function(mps: MatrixProductState):
        cost = 0.0
        for s, weight in cost_function_dict.items():
            s_split = s.split("_")
            if s_split[0] == "S1":
                orbital_idx = int(s_split[1])
                rdm1 = get_one_orbital_rdm(mps, orbital_idx)
                cost += entropy_function(rdm1) * weight
            else:
                orbital_idx1 = int(s_split[1])
                orbital_idx2 = int(s_split[2])
                rdm2 = get_two_orbital_rdm(mps, [orbital_idx1, orbital_idx2])
                cost += entropy_function(rdm2) * weight
        return cost

    return cost_function


def cost_function_dict_to_purity_mpo(
    num_sites: int, cost_function_dict: dict[str, float]
) -> MatrixProductOperator:
    mpos = []
    for s, weight in cost_function_dict.items():
        s_split = s.split("_")
        if s_split[0] == "S1":
            orbital_idx = int(s_split[1])
            spin_orbitals = [2 * orbital_idx - 1, 2 * orbital_idx]
            temp_mpo = MatrixProductOperator.purity_mpo(num_sites, spin_orbitals)
            temp_mpo.multiply_by_constant(weight)
            mpos.append(temp_mpo)
        else:
            orbital_idx1 = int(s_split[1])
            orbital_idx2 = int(s_split[2])
            spin_orbitals = [
                2 * orbital_idx1 - 1,
                2 * orbital_idx1,
                2 * orbital_idx2 - 1,
                2 * orbital_idx2,
            ]
            temp_mpo = MatrixProductOperator.purity_mpo(num_sites, spin_orbitals)
            temp_mpo.multiply_by_constant(weight)
            mpos.append(temp_mpo)

    mpo = mpos[0]
    for next_mpo in mpos[1:]:
        mpo = mpo + next_mpo

    return mpo
