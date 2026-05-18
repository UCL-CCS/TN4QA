from tn4qa.qi_metrics import get_one_orbital_entropy
from tn4qa.mps import MatrixProductState

"""
Benchmark: Active Space Selection Methods
"""

def autocas_selection_ranked_entropy_threshold(mps: MatrixProductState, n_sites: int) -> list:
    """
    Select active space orbitals based on AUTOCAS method.

    Parameters:
    mps (MatrixProductState): The matrix product state.
    n_sites (int): The number of sites (sites are 1-indexed).

    Returns:
    list: Indices of selected active orbitals.
    """
    # Calculate one-orbital entropy for each orbital
    entropies = []
    for i in range(1, n_sites + 1):
        entropy = get_one_orbital_entropy(mps, i)
        entropies.append(entropy)

    # Select top only orbitals whose entropy is at least 10% of the maximum value (AUTOCAS criterion)
    max_entropy = max(entropies)
    top_orbitals = [i for i, entropy in enumerate(entropies) if entropy >= 0.1 * max_entropy]

    return top_orbitals


def autocas_selection_ranked_fixed_number(mps: MatrixProductState, n_sites: int, active_orbitals: int) -> list:
    """
    Select active space orbitals based on AUTOCAS method with fixed number of orbitals.

    Parameters:
    mps (MatrixProductState): The matrix product state.
    n_sites (int): The number of sites (sites are 1-indexed).
    active_orbitals (int): The number of top orbitals to select.

    Returns:
    list: Indices of selected active orbitals.
    """
    # Calculate one-orbital entropy for each orbital
    entropies = []
    for i in range(1, n_sites + 1):
        entropy = get_one_orbital_entropy(mps, i)
        entropies.append(entropy)

    # Select top N orbitals based on entropy
    top_orbitals = sorted(range(len(entropies)), key=lambda i: entropies[i], reverse=True)[:active_orbitals]

    return top_orbitals


def autocas_selection_total_entropy(mps: MatrixProductState, n_sites: int, threshold: float) -> list:
    """
    Select active space orbitals based on AUTOCAS method with fixed total entropy threshold.

    Parameters:
    mps (MatrixProductState): The matrix product state.
    n_sites (int): The number of sites (sites are 1-indexed).
    threshold (float): The entropy threshold for selection.

    Returns:
    list: Indices of selected active orbitals.
    """
    # Calculate one-orbital entropy for each orbital
    entropies = []
    for i in range(1, n_sites + 1):
        entropy = get_one_orbital_entropy(mps, i)
        entropies.append(entropy)

    total_entropy = sum(entropies)
    ranked_orbitals = sorted(range(len(entropies)), key=lambda i: entropies[i], reverse=True)
    desired_entropy = threshold * total_entropy

    top_orbitals = []
    current_entropy = 0
    for i in ranked_orbitals:
        if current_entropy + entropies[i] <= desired_entropy:
            top_orbitals.append(i)
            current_entropy += entropies[i]
        else:
            break

    return top_orbitals