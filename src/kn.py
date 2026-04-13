"""
kn.py — Kim-Nelson (2001) sequential elimination procedure.

Fully sequential, elimination-based ranking & selection for minimization.
Guarantees P(correct selection) >= 1 - alpha whenever the best system's
mean is at least delta better than the second-best.

Uses pairwise difference variances, which automatically capture CRN
covariance.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Any
from config import KN_ALPHA, KN_DELTA, KN_N0, KN_MAX_REPS, N_PORTFOLIOS


def compute_kn_constants(k: int, alpha: float, n0: int) -> Tuple[float, float]:
    """
    Compute the KN continuation constant using Bonferroni correction.

    Parameters
    ----------
    k : int
        Number of systems.
    alpha : float
        Overall Type I error rate.
    n0 : int
        Initial sample size.

    Returns
    -------
    (eta, h_sq) : tuple
        eta: Bonferroni-corrected constant.
        h_sq: 2 * eta * (n0 - 1), used in elimination threshold.
    """
    # Number of pairwise comparisons
    n_pairs = k * (k - 1) // 2
    # Bonferroni: alpha_pair = alpha / n_pairs
    alpha_pair = alpha / n_pairs
    # eta from Rinott-like formulation
    eta = 0.5 * ((2 * alpha_pair) ** (-2.0 / (n0 - 1)) - 1)
    h_sq = 2 * eta * (n0 - 1)
    return eta, h_sq


def kn_procedure(
    simulate_fn: Callable[[int, Dict], float],
    gen_inputs_fn: Callable[[int, int], Dict],
    k: int = N_PORTFOLIOS,
    alpha: float = KN_ALPHA,
    delta: float = KN_DELTA,
    n0: int = KN_N0,
    base_seed: int = 42,
    max_reps: int = KN_MAX_REPS,
) -> Dict[str, Any]:
    """
    Run the Kim-Nelson sequential elimination procedure (minimization).

    Parameters
    ----------
    simulate_fn : callable(portfolio_idx, common_inputs) -> float
        Returns Y_ij for portfolio i given common inputs.
    gen_inputs_fn : callable(base_seed, replication) -> dict
        Generates common random inputs for one replication.
    k : int
        Number of systems (portfolios).
    alpha : float
        Overall significance level; P(CS) >= 1 - alpha.
    delta : float
        Indifference zone (smallest meaningful cost difference).
    n0 : int
        Initial number of replications.
    base_seed : int
        Root seed for CRN.
    max_reps : int
        Safety cap on total replications.

    Returns
    -------
    dict with keys:
        'selected'        : int, index of selected best portfolio
        'survivors'       : set, surviving portfolio indices at termination
        'final_means'     : dict[int, float], sample means of survivors
        'total_reps'      : int, replications consumed
        'elimination_log' : list of (stage, eliminated_idx, its_mean, competitor_mean)
        'all_data'        : dict[int, list], all Y values collected
        'h_sq'            : float, continuation constant
        'pairwise_var'    : dict, S^2_{il} values
    """
    # Step 1: Compute continuation constant
    eta, h_sq = compute_kn_constants(k, alpha, n0)

    # Step 2: Initialization — collect n0 replications from all k systems
    Y = {i: [] for i in range(k)}
    for rep in range(n0):
        inputs = gen_inputs_fn(base_seed, rep)
        for i in range(k):
            Y[i].append(simulate_fn(i, inputs))

    # Pairwise variance of differences
    S_sq = {}
    for i in range(k):
        for l in range(k):
            if i != l:
                D = np.array([Y[i][j] - Y[l][j] for j in range(n0)])
                S_sq[(i, l)] = np.var(D, ddof=1)

    # Initialize survivors and log
    I = set(range(k))
    r = n0
    elimination_log = []

    # Step 3-4: Sequential screening
    while len(I) > 1 and r < max_reps:
        # Compute running means for survivors
        Y_bar = {i: np.mean(Y[i][:r]) for i in I}

        # Screen: eliminate portfolios whose mean is too high
        to_remove = set()
        for i in list(I):
            if i in to_remove:
                continue
            for l in list(I):
                if i == l or l in to_remove:
                    continue
                # Elimination threshold (shrinks to zero over time)
                W = max(0, (delta / (2 * r)) *
                        (h_sq * S_sq[(i, l)] / delta**2 - r))
                # Eliminate i if its mean exceeds l's by more than threshold
                if Y_bar[i] > Y_bar[l] + W:
                    to_remove.add(i)
                    elimination_log.append((r, i, Y_bar[i], Y_bar[l], l))
                    break

        I -= to_remove

        if len(I) <= 1:
            break

        # Check terminal condition
        max_N = max(int(h_sq * S_sq[(i, l)] / delta**2) + 1
                    for i in I for l in I if i != l)
        if r >= max_N:
            break

        # Take one more replication from each survivor
        r += 1
        inputs = gen_inputs_fn(base_seed, r - 1)
        for i in I:
            Y[i].append(simulate_fn(i, inputs))
        # Also generate (but discard) for eliminated systems to maintain CRN sync
        # (not strictly necessary since SeedSequence handles this, but explicit)

    # Select survivor with lowest mean
    best = min(I, key=lambda i: np.mean(Y[i][:r]))

    return {
        'selected': best,
        'survivors': I,
        'final_means': {i: np.mean(Y[i][:r]) for i in I},
        'total_reps': r,
        'elimination_log': elimination_log,
        'all_data': Y,
        'h_sq': h_sq,
        'pairwise_var': S_sq,
    }


def format_kn_results(results: Dict, portfolios: list) -> str:
    """Format KN results as a readable summary."""
    lines = []
    lines.append(f"KN Procedure Results (k={len(portfolios)} systems)")
    lines.append(f"{'='*60}")
    lines.append(f"Selected best: {portfolios[results['selected']].label}")
    lines.append(f"Total replications: {results['total_reps']}")
    lines.append(f"Survivors at termination: {len(results['survivors'])}")
    lines.append(f"")

    if results['elimination_log']:
        lines.append("Elimination timeline:")
        lines.append(f"  {'Stage':>6} {'Eliminated':<20} {'Its Mean ($M)':>14} "
                     f"{'vs':>4} {'Competitor':<20} {'Comp Mean ($M)':>14}")
        lines.append(f"  {'-'*80}")
        for stage, elim, elim_mean, comp_mean, comp_idx in results['elimination_log']:
            lines.append(
                f"  {stage:>6} {portfolios[elim].label:<20} "
                f"{elim_mean/1e6:>14,.1f} {'vs':>4} "
                f"{portfolios[comp_idx].label:<20} {comp_mean/1e6:>14,.1f}"
            )
    else:
        lines.append("No eliminations occurred (all within indifference zone).")

    lines.append(f"")
    lines.append("Surviving portfolio means:")
    for idx in sorted(results['survivors']):
        mean_y = results['final_means'][idx]
        lines.append(f"  {portfolios[idx].label}: ${mean_y/1e6:,.1f}M")

    return "\n".join(lines)


if __name__ == "__main__":
    from config import PORTFOLIOS
    from crn import generate_common_inputs
    from dispatch import simulate_replication

    def sim_fn(i, inputs):
        return simulate_replication(PORTFOLIOS[i], inputs)[3]  # Y_ij

    print(f"Running KN with k={N_PORTFOLIOS}, alpha={KN_ALPHA}, "
          f"delta=${KN_DELTA/1e6:.0f}M, n0={KN_N0}")
    eta, h_sq = compute_kn_constants(N_PORTFOLIOS, KN_ALPHA, KN_N0)
    print(f"eta={eta:.4f}, h^2={h_sq:.2f}")
    print()

    results = kn_procedure(sim_fn, generate_common_inputs)
    print(format_kn_results(results, PORTFOLIOS))
