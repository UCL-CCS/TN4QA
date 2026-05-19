from tn4qa.qi_metrics import get_one_orbital_entropy
from tn4qa.mps import MatrixProductState

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
