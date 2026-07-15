import itertools

import numpy as np

from tn4qa.mps import MatrixProductState
from tn4qa.qi_metrics import get_one_orbital_entropy
from tn4qa.tn_methods.entanglement_feature import (
    build_entanglement_feature,
    contract_ef_bitstring,
)

"""
Benchmark: Active Space Selection Methods
"""


def _orbital_entropies(mps: MatrixProductState, n_sites: int) -> list[float]:
    """Compute one-orbital entropies for orbitals 1..n_sites."""
    return [get_one_orbital_entropy(mps, i) for i in range(1, n_sites + 1)]


def autocas_selection_ranked_entropy_threshold(
    mps: MatrixProductState, n_sites: int
) -> list[int]:
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

    return [
        idx for idx, entropy in enumerate(entropies) if entropy >= 0.1 * max_entropy
    ]


def autocas_selection_ranked_fixed_number(
    mps: MatrixProductState, active_orbitals: int
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
    n_spin_orbs = mps.num_sites
    n_spatial_orbs = n_spin_orbs // 2
    entropies = _orbital_entropies(mps, n_spatial_orbs)
    active_orbitals = min(active_orbitals, len(entropies))

    return sorted(range(len(entropies)), key=lambda i: entropies[i], reverse=True)[
        :active_orbitals
    ]


def autocas_selection_total_entropy(
    mps: MatrixProductState, n_sites: int, threshold: float = 0.1
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
    ranked_orbitals = sorted(
        range(len(entropies)), key=lambda i: entropies[i], reverse=True
    )

    selected_orbitals = []
    current_entropy = 0.0
    for idx in ranked_orbitals:
        selected_orbitals.append(idx)
        current_entropy += entropies[idx]
        if current_entropy >= desired_entropy:
            break

    return selected_orbitals


def ef_subset_entropy(ef_mps, subset: list[int], n_spin_orbs: int) -> float:
    """
    Compute the Renyi-2 entropy between a subset of orbitals and the rest using the entanglement feature.
    subset: list of orbital indices
    Returns: Renyi-2 entropy (float)
    """
    # contract EF with bitstrings
    bitstring = [0] * n_spin_orbs
    for i in subset:
        if 2 * i + 1 < n_spin_orbs:
            bitstring[2 * i] = 1
            bitstring[2 * i + 1] = 1

    R2 = contract_ef_bitstring(ef_mps, bitstring)
    ef_entropy = -np.log2(R2)
    return ef_entropy


def ef_active_space_brute_force(mps: MatrixProductState, n_sites: int) -> list[int]:
    """
    Select active space based on the entanglement feature.
    """
    if n_sites <= 0:
        return []

    total_orbitals = mps.num_sites // 2
    ef_mps = build_entanglement_feature(mps)

    best_subset = max(
        itertools.combinations(range(total_orbitals), n_sites),
        key=lambda subset: ef_subset_entropy(ef_mps, subset, mps.num_sites),
    )
    return list(best_subset)


def ef_active_space_greedy(mps: MatrixProductState, active_orbitals: int) -> list[int]:
    """
    Greedy selection of active space based on the entanglement feature.
    """
    selected_orbitals = []
    n_spin_orbs = mps.num_sites
    n_spatial_orbs = n_spin_orbs // 2
    current_bitstring = [0] * n_spin_orbs
    remaining_orbitals = set(range(n_spatial_orbs))
    # print(remaining_orbitals)
    entropies = _orbital_entropies(mps, n_spatial_orbs)
    entropies = [float(i) for i in entropies]
    sorted_entropies = sorted(
        range(len(entropies)), key=lambda i: entropies[i], reverse=True
    )
    # print(f"Orbital entropies: {entropies}")
    best_initial_orbital = sorted_entropies[0]
    # print(f"Best initial orbital: {best_initial_orbital} with entropy {entropies[best_initial_orbital]:.4f}")
    selected_orbitals.append(best_initial_orbital)
    remaining_orbitals.remove(best_initial_orbital)
    init_spin_index = 2 * best_initial_orbital
    current_bitstring[init_spin_index] = 1
    current_bitstring[init_spin_index + 1] = 1

    while len(selected_orbitals) < active_orbitals and remaining_orbitals:
        best_orbital = None
        best_entropy = -float("inf")

        for orbital in remaining_orbitals:
            candidate_subset = selected_orbitals + [orbital]
            ef_mps = build_entanglement_feature(mps)
            entropy = ef_subset_entropy(ef_mps, candidate_subset, n_spin_orbs)
            if entropy > best_entropy:
                best_entropy = entropy
                best_orbital = orbital

        if best_orbital is not None:
            selected_orbitals.append(best_orbital)
            remaining_orbitals.remove(best_orbital)
        else:
            break

    return selected_orbitals


def ef_active_space_sample(
    mps: MatrixProductState, n_sites: int, n_samples: int = 10000, seed: int = None
) -> list[int]:
    """
    Sample bitstrings from the EF MPS and rank spatial orbitals by how often
    they appear as active (bit=1) in high-weight samples. Selects the top
    n_sites orbitals.
    """
    n_spin_orbs = mps.num_sites
    n_spatial_orbs = n_spin_orbs // 2
    orbital_counts = np.zeros(n_spatial_orbs, dtype=float)

    ef_mps = build_entanglement_feature(mps)
    bitstring_counts = ef_mps.sample_bitstrings(num_samples=n_samples, seed=seed)

    for bitstring_str, count in bitstring_counts.items():
        valid = True
        active_indices = []

        for orb in range(n_spatial_orbs):
            start = 2 * orb
            pair = bitstring_str[start : start + 2]
            if pair == "11":
                active_indices.append(orb)
            elif pair == "00":
                continue
            else:
                valid = False
                break

        if not valid:
            continue

        for orb in active_indices:
            orbital_counts[orb] += count

    n_sites = int(n_sites)
    if orbital_counts.sum() == 0:
        return list(range(min(n_sites, n_spatial_orbs)))

    top_indices = np.argsort(orbital_counts)[:n_sites]
    return [int(idx) for idx in top_indices]


def ef_active_space_greedy_electic_boogaloo(
    mps: MatrixProductState, active_orbitals: int, k: int
) -> list[int]:
    """
    Greedy selection of active space based on the entanglement feature.
    Starts with the orbital of highest one-orbital entropy and adds orbitals that maximize the Renyi-2 entropy of the selected subset.
    Until it reaches k and then it restarts with the next highest one-orbital entropy orbital not in the selected subset.
    Repeats until active_orbitals are selected.
    """
    selected_orbitals = []
    n_spin_orbs = mps.num_sites
    n_spatial_orbs = n_spin_orbs // 2
    current_bitstring = [0] * n_spin_orbs
    remaining_orbitals = set(range(n_spatial_orbs))
    # print(remaining_orbitals)
    entropies = _orbital_entropies(mps, n_spatial_orbs)
    entropies = [float(i) for i in entropies]
    sorted_entropies = sorted(
        range(len(entropies)), key=lambda i: entropies[i], reverse=True
    )
    # print(f"Orbital entropies: {entropies}")
    best_initial_orbital = sorted_entropies[0]
    # print(f"Best initial orbital: {best_initial_orbital} with entropy {entropies[best_initial_orbital]:.4f}")
    selected_orbitals.append(best_initial_orbital)
    remaining_orbitals.remove(best_initial_orbital)
    current_set = [best_initial_orbital]
    init_spin_index = 2 * best_initial_orbital
    current_bitstring[init_spin_index] = 1
    current_bitstring[init_spin_index + 1] = 1

    while len(selected_orbitals) < active_orbitals and remaining_orbitals:
        if len(selected_orbitals) % k == 0:
            current_set = []
            restarting = False
            # Restart with the next highest one-orbital entropy orbital not in the selected subset
            for orbital in sorted_entropies:
                if orbital not in selected_orbitals:
                    selected_orbitals.append(orbital)
                    remaining_orbitals.remove(orbital)
                    current_set.append(orbital)
                    init_spin_index = 2 * orbital
                    current_bitstring[init_spin_index] = 1
                    current_bitstring[init_spin_index + 1] = 1
                    restarting = True
                    break
            if restarting:
                continue
            else:
                break

        best_orbital = None
        best_entropy = -float("inf")

        for orbital in remaining_orbitals:
            candidate_subset = current_set + [orbital]
            # candidate_subset = selected_orbitals + [orbital]
            ef_mps = build_entanglement_feature(mps)
            entropy = ef_subset_entropy(ef_mps, candidate_subset, n_spin_orbs)
            if entropy > best_entropy:
                best_entropy = entropy
                best_orbital = orbital

        if best_orbital is not None:
            selected_orbitals.append(best_orbital)
            remaining_orbitals.remove(best_orbital)
            current_set.append(best_orbital)
        else:
            break

    return selected_orbitals


def ef_active_space_mcmc(
    mps: MatrixProductState,
    n_active: int,
    n_steps: int = 5000,
    beta_start: float = 3.0,
    beta_end: float = 10.0,
    seed: int = None,
) -> list[int]:
    """
    Select active space by simulated annealing on the Rényi-2 entropy S(subset),
    using the Entanglement Feature as an entropy oracle.
    The objective is to find the subset A of size n_active that maximises
    S(A), the Rényi-2 entropy of the bipartition A | complement(A).
    We use Metropolis-Hastings with a fixed-size swap proposal: at each step,
    remove one orbital from the current active space and add one from outside,
    keeping |A| = n_active fixed throughout. The acceptance probability is
        min(1, exp(beta * (S(x') - S(x))))
    where beta is annealed linearly from beta_start to beta_end over the run.
    At low beta the chain explores broadly; at high beta it concentrates near
    the maximum.
    This is qualitatively different from greedy: greedy is a deterministic
    hill-climb that commits to each orbital permanently and cannot backtrack.
    MCMC can escape local optima by occasionally accepting worse moves early
    in the run, and the annealing schedule systematically reduces this
    willingness as the chain converges.
    Parameters

    ----------

    mps        : reference MPS (DMRG solution)

    n_active   : number of spatial orbitals to select (held fixed throughout)

    n_steps    : number of MCMC swap proposals

    beta_start : initial inverse temperature (low = more exploratory)

    beta_end   : final inverse temperature (high = more greedy)

    seed       : RNG seed for reproducibility

    Returns

    -------

    list[int] : selected orbital indices (0-based, spatial), the best subset

                seen across the entire chain (not just the final state)

    """

    rng = np.random.default_rng(seed)
    n_spin_orbs = mps.num_sites
    n_spatial_orbs = n_spin_orbs // 2
    ef_mps = build_entanglement_feature(mps)
    # Initialise with the n_active orbitals of highest single-orbital entropy.
    # This is the autoCAS starting point, which is a reasonable warm start
    # (much better than random, avoids wasting early steps on obviously bad states).
    from tn4qa.qi_metrics import get_one_orbital_entropy

    single_entropies = [
        get_one_orbital_entropy(mps, i) for i in range(1, n_spatial_orbs + 1)
    ]
    average_entropy = np.mean(single_entropies)
    twice_mean = 2 * average_entropy
    beta_start = 1 / twice_mean * np.log(0.5)
    beta_end = 10 * beta_start
    init_ranked = sorted(
        range(n_spatial_orbs), key=lambda i: single_entropies[i], reverse=True
    )
    current = set(init_ranked[:n_active])
    current_entropy = ef_subset_entropy(ef_mps, list(current), n_spin_orbs)
    best_subset = set(current)
    best_entropy = current_entropy
    betas = np.linspace(beta_start, beta_end, n_steps)
    for step in range(n_steps):
        beta = betas[step]
        # Swap proposal: remove one orbital from active, add one from inactive
        # Make it rare for high entropy orbitals to be removed
        weights = np.array([single_entropies[i] for i in current])
        weights = np.exp(-weights / np.max(weights))  # Inverse weighting
        weights /= np.sum(weights)
        remove = rng.choice(list(current), p=weights)
        inactive = list(set(range(n_spatial_orbs)) - current)
        add = int(rng.choice(inactive))
        proposed = (current - {remove}) | {add}
        proposed_entropy = ef_subset_entropy(ef_mps, list(proposed), n_spin_orbs)
        # Metropolis acceptance
        delta = proposed_entropy - current_entropy
        if delta > 0 or rng.random() < np.exp(beta * delta):
            current = proposed
            current_entropy = proposed_entropy
        # Track the best subset seen anywhere in the chain
        if current_entropy > best_entropy:
            best_entropy = current_entropy
            best_subset = set(current)
    return sorted(int(i) for i in best_subset)
