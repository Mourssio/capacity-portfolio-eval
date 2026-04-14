"""
capital_sensitivity.py -- One-at-a-time sensitivity of portfolio rankings
to capital cost assumptions.

Capital costs feed only F_i (fixed cost). C_ij and U_ij are unaffected.
So this analysis is pure post-processing on existing Mean-Y decomposition:
  new_F_i = base_F_i + delta_F
  new_Y   = old_Y - old_F + new_F

We vary each cost parameter by +/- 30% around its baseline and record how
the top-4 ranking responds. Baselines are informed by NREL ATB 2023 and
IESO Annual Planning Outlook 2023.
"""

import numpy as np
from config import (
    PORTFOLIOS, FIRM_SUBTYPES, RENEWABLE_SPLIT, CAPITAL_COST,
)
from crn import generate_common_inputs
from dispatch import simulate_replication


# Baseline cost components per portfolio (in $) attributable to each parameter.
# Structure: for each portfolio, the MW quantity that each capital cost multiplies.
def _mw_per_cost(p):
    """
    Return dict mapping cost parameter -> installed MW that the cost multiplies,
    for portfolio p.
    """
    out = {name: info["share"] * p.firm_mw for name, info in FIRM_SUBTYPES.items()}
    out["wind"] = p.wind_mw
    out["solar"] = p.solar_mw
    out["storage"] = p.storage_mw
    return out


def _fixed_cost_from_vector(p, cost_vector):
    """Recompute F_i using a perturbed cost vector (in $/kW-yr)."""
    mw = _mw_per_cost(p)
    total = 0.0
    for name, mw_val in mw.items():
        total += mw_val * 1000 * cost_vector[name]
    return total


def run_capital_sensitivity(top_k_indices, mean_Y_base, mean_F_base,
                             perturbations=(-0.30, -0.15, 0.0, 0.15, 0.30),
                             verbose=True):
    """
    Sweep each cost parameter through +/- 30% and recompute Mean Y and rank.

    Parameters
    ----------
    top_k_indices : list[int]
        Portfolio indices to evaluate (in rank order).
    mean_Y_base : dict[int, float]
        Baseline mean Y per portfolio from the main CRN run.
    mean_F_base : dict[int, float]
        Baseline fixed cost per portfolio (deterministic).

    Returns
    -------
    results : dict
        Per-parameter ranking changes and max effect magnitudes.
    """
    params = list(CAPITAL_COST.keys())
    results = {}

    for param in params:
        base_cost = CAPITAL_COST[param]
        rank_changes = []
        max_best_delta = 0.0
        min_gap = float("inf")  # Smallest gap across all perturbations
        for pct in perturbations:
            perturbed = dict(CAPITAL_COST)
            perturbed[param] = base_cost * (1.0 + pct)

            new_means = {}
            for i in top_k_indices:
                p = PORTFOLIOS[i]
                new_F = _fixed_cost_from_vector(p, perturbed)
                new_Y = mean_Y_base[i] - mean_F_base[i] + new_F
                new_means[i] = new_Y

            ranked = sorted(new_means.items(), key=lambda kv: kv[1])
            best = ranked[0][0]
            gap = ranked[1][1] - ranked[0][1]
            rank_changes.append({
                "pct": pct, "best": best, "means": new_means, "gap": gap,
            })
            if gap < min_gap:
                min_gap = gap

        unique_bests = set(rc["best"] for rc in rank_changes)
        results[param] = {
            "base_cost": base_cost,
            "unique_bests": unique_bests,
            "rank_stable": len(unique_bests) == 1,
            "min_gap_over_sweep": min_gap,
            "sweep": rank_changes,
        }
        if verbose:
            print(f"  {param:<12} base=${base_cost:>4}/kW-yr  "
                  f"sweep ±30%  min gap=${min_gap/1e6:>8,.1f}M  "
                  f"stable best: {results[param]['rank_stable']}  "
                  f"unique bests: {len(unique_bests)}")
    return results


def _collect_baseline_from_pipeline(n_reps=200):
    """
    Run a small CRN evaluation of the top-4 to get baseline Mean Y and F.
    Uses the same top-4 identified in the full pipeline (P04, P10, P03, P09).
    """
    # Known top-4 from the existing KN + extended run
    top_4_labels = ["P04", "P10", "P03", "P09"]
    top_k = []
    for label_prefix in top_4_labels:
        idx = next(p.pid for p in PORTFOLIOS if p.label.startswith(label_prefix))
        top_k.append(idx)

    Y_sums = {i: [] for i in top_k}
    F_vals = {i: PORTFOLIOS[i].annualized_fixed_cost() for i in top_k}
    for rep in range(n_reps):
        inp = generate_common_inputs(base_seed=42, replication=rep)
        for i in top_k:
            _, _, _, Y = simulate_replication(PORTFOLIOS[i], inp)
            Y_sums[i].append(Y)
    mean_Y = {i: float(np.mean(Y_sums[i])) for i in top_k}
    return top_k, mean_Y, F_vals


if __name__ == "__main__":
    print("=" * 70)
    print("CAPITAL COST SENSITIVITY (+/- 30% one-at-a-time)")
    print("=" * 70)
    print("Collecting baseline over 200 CRN replications...")
    top_k, mean_Y, mean_F = _collect_baseline_from_pipeline(n_reps=200)
    print(f"  Baseline top-4 mean Y:")
    for i in top_k:
        print(f"    {PORTFOLIOS[i].label:<22} ${mean_Y[i]/1e6:>10,.1f}M")
    print()
    results = run_capital_sensitivity(top_k, mean_Y, mean_F)

    # Summary: is ranking stable across all one-at-a-time ±30% sweeps?
    all_stable = all(r["rank_stable"] for r in results.values())
    print()
    print("=" * 70)
    print(f"RANKING STABILITY: "
          f"{'STABLE' if all_stable else 'SENSITIVE'} "
          f"(best portfolio {'unchanged' if all_stable else 'changes'} across all sweeps)")
    print("=" * 70)
