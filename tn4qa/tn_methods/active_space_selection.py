from tn4qa.qi_metrics import get_one_orbital_entropy
from tn4qa.mps import MatrixProductState

"""
Benchmark: Active Space Selection Methods
"""

# AUTOCAS
def autocas_selection(mps: MatrixProductState, n_sites: int) -> list:
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