import itertools

from tn4qa.qi_metrics import get_one_orbital_entropy
from tn4qa.mps import MatrixProductState
from tn4qa.tn_methods.entanglement_feature import build_entanglement_feature, contract_ef_bitstring
import numpy as np


"""
Benchmark: Active Space Selection Methods
"""


def _orbital_entropies(mps: MatrixProductState, n_sites: int) -> list[float]:
    """Compute one-orbital entropies for orbitals 1..n_sites."""
    return [get_one_orbital_entropy(mps, i) for i in range(1, n_sites + 1)]


def autocas_selection_ranked_entropy_threshold(mps: MatrixProductState, n_sites: int) -> list[int]:
    """
    Select orbital indices whose one-orbital entropy is at least 10% of the
    maximum orbital entropy.

    Parameters:
        mps: MatrixProductState
        n_sites: int
            Number of spatial orbitals.

    Returns:
        list[int]: Selected orbital indices in 0-based notation.
    """
    entropies = _orbital_entropies(mps, n_sites)
    max_entropy = max(entropies)

    if max_entropy <= 0:
        return []

    return [idx for idx, entropy in enumerate(entropies) if entropy >= 0.1 * max_entropy]


def autocas_selection_ranked_fixed_number(
    mps: MatrixProductState, n_sites: int, active_orbitals: int
) -> list[int]:
    """
    Select the top N orbitals by one-orbital entropy.

    Parameters:
        mps: MatrixProductState
        n_sites: int
            Number of spatial orbitals.
        active_orbitals: int
            Number of orbitals to select.

    Returns:
        list[int]: Selected orbital indices in 0-based notation.
    """
    if active_orbitals <= 0:
        return []

    entropies = _orbital_entropies(mps, n_sites)
    active_orbitals = min(active_orbitals, len(entropies))

    return sorted(range(len(entropies)), key=lambda i: entropies[i], reverse=True)[:active_orbitals]


def autocas_selection_total_entropy(
    mps: MatrixProductState, n_sites: int, threshold: float
) -> list[int]:
    """
    Select orbitals until the cumulative entropy reaches the requested threshold
    fraction of the total entropy.

    Parameters:
        mps: MatrixProductState
        n_sites: int
            Number of spatial orbitals.
        threshold: float
            Fraction of total entropy to capture.

    Returns:
        list[int]: Selected orbital indices in 0-based notation.
    """
    if threshold <= 0:
        return []
    if threshold >= 1:
        return list(range(n_sites))

    entropies = _orbital_entropies(mps, n_sites)
    total_entropy = sum(entropies)

    if total_entropy <= 0:
        return []

    desired_entropy = threshold * total_entropy
    ranked_orbitals = sorted(range(len(entropies)), key=lambda i: entropies[i], reverse=True)

    selected_orbitals = []
    current_entropy = 0.0
    for idx in ranked_orbitals:
        selected_orbitals.append(idx)
        current_entropy += entropies[idx]
        if current_entropy >= desired_entropy:
            break

    return selected_orbitals

def ef_subset_entropy(mps: MatrixProductState, subset: list[int]) -> float:
    """
    Compute the Renyi-2 entropy between a subset of orbitals and the rest using the entanglement feature.
    subset: list of orbital indices (1-based)
    Returns: Renyi-2 entropy (float)
    """
    ef_mps = build_entanglement_feature(mps)
    n_sites = ef_mps.num_sites
    # contract EF with bitstrings
    bitstring = [0] * n_sites
    for i in subset:
        bitstring[2 * i] = 1
        bitstring[2 * i + 1] = 1
    
    R2 = contract_ef_bitstring(ef_mps, bitstring)
    ef_entropy = -np.log2(R2)
    return ef_entropy

def ef_active_space(mps: MatrixProductState, n_sites: int) -> list[int]:
    """
    Select active space based on the entanglement feature.
    For each subset of n_sites orbitals, compute the Renyi-2 entropy using the entanglement feature,
    and select orbitals with entropy above a certain threshold.
    Parameters:
        mps: MatrixProductState
        n_sites: int
            Number of spatial orbitals.

    Returns:
        list[int]: Selected orbital indices in 0-based notation.
    """
    # Select subsets of orbitals with n_sites orbitals
    total_orbitals = mps.num_sites // 2
    subsets = itertools.combinations(range(total_orbitals), n_sites)
    subset_entropies = []
    for subset in subsets:
        entropy = ef_subset_entropy(mps, subset)
        subset_entropies.append((subset, entropy))

    # Select subset with highest entropy
    subset_entropies.sort(key=lambda x: x[1], reverse=True)
    best_subset = subset_entropies[0][0]
    return list(best_subset)

