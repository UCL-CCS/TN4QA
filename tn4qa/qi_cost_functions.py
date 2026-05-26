from typing import Callable

import numpy as np
from numpy import ndarray

from .mpo import MatrixProductOperator
from .mps import MatrixProductState
from .qi_metrics import (
    get_all_mutual_information,
    get_mutual_information,
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
        s[f"S1_{i + 1}"] = 1.0
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
        s[f"S1_{i + 1}"] = s.get(f"S1_{i + 1}", 0) + 1.0
        for j in range(i + 1, num_orbitals):
            s[f"S1_{i + 1}"] = s.get(f"S1_{i + 1}", 0) + 1.0
            s[f"S1_{j + 1}"] = s.get(f"S1_{j + 1}", 0) + 1.0
            s[f"S2_{i + 1}_{j + 1}"] = s.get(f"S2_{i + 1}_{j + 1}", 0) - 1.0
    return s


def cost_mutual_info_active_inactive(
    mps: MatrixProductState, active_orbs: list[int]
) -> float:
    """
    Calculates the mutual information between the active and inactive regions
    """
    n_orbs = mps.num_sites // 2  # Number of orbitals
    active_set = set(active_orbs)
    inactive_orbs = [
        i for i in range(1, n_orbs + 1) if i not in active_set
    ]  # Not sure whether to do it like this or to give inactive orbs as an arg
    mi = 0
    for i in active_orbs:
        for j in inactive_orbs:
            mi += get_mutual_information(mps, i, j)
    return mi


def cost_mutual_info_active_inactive_dict(
    num_orbs: int, active_orbs: list[int]
) -> dict:
    active_set = set(active_orbs)
    inactive_orbs = [
        i for i in range(1, num_orbs + 1) if i not in active_set
    ]  # similar comment as above
    s = {}
    for i in active_orbs:
        for j in inactive_orbs:
            s[f"S1_{i}"] = s.get(f"S1_{i}", 0) + 1.0
            s[f"S1_{j}"] = s.get(f"S1_{j}", 0) + 1.0
            s[f"S2_{i}_{j}"] = s.get(f"S2_{i}_{j}", 0) - 1.0
    return s


def cost_mutual_info_active(mps: MatrixProductState, active_orbs: list[int]):
    mi = 0
    for i in range(len(active_orbs)):
        for j in range(i + 1, len(active_orbs)):
            mi -= get_mutual_information(mps, active_orbs[i], active_orbs[j])
    return mi


def cost_mutual_info_active_dict(num_orbs: int, active_orbs: list[int]) -> dict:
    s = {}
    for idx1 in range(len(active_orbs)):
        for idx2 in range(idx1 + 1, len(active_orbs)):
            i, j = active_orbs[idx1], active_orbs[idx2]
            s[f"S1_{i}"] = s.get(f"S1_{i}", 0) - 1.0
            s[f"S1_{j}"] = s.get(f"S1_{j}", 0) - 1.0
            s[f"S2_{i}_{j}"] = s.get(f"S2_{i}_{j}", 0) + 1.0
    return s


def cost_entropy_active(mps: MatrixProductState, active_orbs: list[int]):
    total = 0
    for i in active_orbs:
        e = get_one_orbital_entropy(mps, i)
        total -= e
    return total


def cost_entropy_active_dict(num_orbs: int, active_orbs: list[int]) -> dict:
    s = {}
    for i in active_orbs:
        s[f"S1_{i}"] = s.get(f"S1_{i}", 0) - 1.0

    return s


def cost_minimise_environment_entropy(
    mps: MatrixProductState, active_orbs: list[int]
) -> float:
    num_orbs = mps.num_sites // 2
    active_set = set(active_orbs)
    inactive_orbs = [i for i in range(1, num_orbs + 1) if i not in active_set]
    total = 0
    for i in inactive_orbs:
        e = get_one_orbital_entropy(mps, i)
        total += e
    return total


def cost_minimise_environment_entropy_dict(
    num_orbs: int, active_orbs: list[int]
) -> float:
    active_set = set(active_orbs)
    inactive_orbs = [i for i in range(1, num_orbs + 1) if i not in active_set]
    s = {}
    for i in inactive_orbs:
        s[f"S1_{i}"] = s.get(f"S1_{i}", 0) + 1.0

    return s


def cost_balanced(mps: MatrixProductState, active_orbs: list[int], alpha: float = 0.5):
    n_orbs = mps.num_sites // 2  # Number of orbitals
    active_set = set(active_orbs)
    inactive_orbs = [
        i for i in range(1, n_orbs + 1) if i not in active_set
    ]  # Not sure whether to do it like this or to give inactive orbs as an arg
    mi = 0
    for i in active_orbs:
        for j in inactive_orbs:
            mi += get_mutual_information(mps, i, j)

    e = 0
    for i in inactive_orbs:
        e += get_one_orbital_entropy(mps, i)

    return mi + alpha * e


def cost_balanced_dict(
    num_orbs: int, active_orbs: list[int], alpha: float = 0.5
) -> dict:
    active_set = set(active_orbs)
    inactive_orbs = [
        i for i in range(1, num_orbs + 1) if i not in active_set
    ]  # similar comment as above
    s = {}
    for i in active_orbs:
        for j in inactive_orbs:
            s[f"S1_{i}"] = s.get(f"S1_{i}", 0) + 1.0
            s[f"S1_{j}"] = s.get(f"S1_{j}", 0) + 1.0
            s[f"S2_{i}_{j}"] = s.get(f"S2_{i}_{j}", 0) - 1.0
    for i in inactive_orbs:
        s[f"S1_{i}"] = s.get(f"S1_{i}", 0) + alpha
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
            s[f"S1_{i + 1}"] = s.get(f"S1_{i + 1}", 0) + (distance**decay_power)
            s[f"S1_{j + 1}"] = s.get(f"S1_{j + 1}", 0) + (distance**decay_power)
            s[f"S2_{i + 1}_{j + 1}"] = s.get(f"S2_{i + 1}_{j + 1}", 0) - (
                distance**decay_power
            )
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
            return cost_mutual_info_decay_dict(
                num_orbitals=num_orbitals, decay_power=decay_power
            )
        case "cost_mutual_info_active_inactive":
            num_orbitals = function_params["num_orbitals"]
            active_orbs = function_params["active_orbs"]
            # active_orbs = list(range(num_active))
            return cost_mutual_info_active_inactive_dict(
                num_orbs=num_orbitals, active_orbs=active_orbs
            )
        case "cost_mutual_info_active":
            num_orbitals = function_params["num_orbitals"]
            active_orbs = function_params["active_orbs"]
            # active_orbs = list(range(num_active))
            return cost_mutual_info_active_dict(
                num_orbs=num_orbitals, active_orbs=active_orbs
            )
        case "cost_entropy_active":
            num_orbitals = function_params["num_orbitals"]
            active_orbs = function_params["active_orbs"]
            return cost_entropy_active_dict(
                num_orbs=num_orbitals, active_orbs=active_orbs
            )
        case "cost_minimise_environment_entropy":
            num_orbitals = function_params["num_orbitals"]
            active_orbs = function_params["active_orbs"]
            return cost_minimise_environment_entropy_dict(
                num_orbs=num_orbitals, active_orbs=active_orbs
            )
        case "cost_balanced":
            num_orbitals = function_params["num_orbitals"]
            active_orbs = function_params["active_orbs"]
            return cost_balanced_dict(
                num_orbs=num_orbitals, active_orbs=active_orbs, alpha=0.5
            )
        case _:
            raise ValueError
    return


def calculate_purity(density_matrix: ndarray) -> float:
    rho_squared = density_matrix @ density_matrix
    purity = np.trace(rho_squared)
    return purity


def calculate_pseudo_entropy(density_matrix: ndarray) -> float:
    p = calculate_purity(density_matrix)
    return 1 - p


def cost_function_dict_to_callable(
    cost_function_dict: dict[str, float],
    entropy_function: Callable[[ndarray], float],
) -> Callable[[MatrixProductState], float]:
    def cost_function(mps: MatrixProductState):
        cost = 0.0
        for s, weight in cost_function_dict.items():
            # print(s, weight)
            s_split = s.split("_")
            if s_split[0] == "S1":
                orbital_idx = int(s_split[1])
                rdm1 = get_one_orbital_rdm(mps, orbital_idx)
                cost += entropy_function(rdm1) * weight
                print(entropy_function(rdm1) * weight)
            else:
                orbital_idx1 = int(s_split[1])
                orbital_idx2 = int(s_split[2])
                rdm2 = get_two_orbital_rdm(mps, [orbital_idx1, orbital_idx2])
                cost += entropy_function(rdm2) * weight
                print(entropy_function(rdm2) * weight)
        return cost

    return cost_function


def cost_function_dict_to_purity_mpo(
    num_sites: int,
    cost_function_dict: dict[str, float],
    max_bond: int | None = None,
) -> MatrixProductOperator:
    mpos = []
    for s, weight in cost_function_dict.items():
        s_split = s.split("_")
        if s_split[0] == "S1":
            orbital_idx = int(s_split[1])
            spin_orbitals = [2 * orbital_idx - 1, 2 * orbital_idx]
            id_mpo = MatrixProductOperator.identity_mpo(2 * num_sites)
            temp_mpo = MatrixProductOperator.purity_mpo_direct(
                num_sites, tuple(spin_orbitals), max_bond=max_bond
            )
            diff = id_mpo - temp_mpo
            diff.multiply_by_constant(weight)
            mpos.append(diff)
        else:
            orbital_idx1 = int(s_split[1])
            orbital_idx2 = int(s_split[2])
            spin_orbitals = [
                2 * orbital_idx1 - 1,
                2 * orbital_idx1,
                2 * orbital_idx2 - 1,
                2 * orbital_idx2,
            ]
            id_mpo = MatrixProductOperator.identity_mpo(2 * num_sites)
            temp_mpo = MatrixProductOperator.purity_mpo_direct(
                num_sites, tuple(spin_orbitals), max_bond=max_bond
            )
            diff = id_mpo - temp_mpo
            diff.multiply_by_constant(weight)
            mpos.append(diff)

    def _tree_sum_and_compress(mpos, max_bond):
        """O(log n) depth rather than O(n), keeps intermediate bond dims small."""
        while len(mpos) > 1:
            next_level = []
            for i in range(0, len(mpos), 2):
                if i + 1 < len(mpos):
                    combined = mpos[i] + mpos[i + 1]
                    if combined.bond_dimension > max_bond:
                        combined.compress(max_bond)
                    next_level.append(combined)
                else:
                    next_level.append(mpos[i])
            mpos = next_level
        return mpos[0]

    mpo = _tree_sum_and_compress(mpos, max_bond)

    final_mpo = MatrixProductOperator.from_arrays([m.data for m in mpo.tensors])

    return final_mpo
