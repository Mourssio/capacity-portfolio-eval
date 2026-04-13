"""
experiment.py — Experimental analysis components.

A. KN selection (handled by kn.py)
B. Deterministic Benchmark Gap (DBG)
C. Sensitivity analysis (2^3 factorial on top-k survivors)
D. Confidence intervals with Bonferroni correction
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
from itertools import combinations
from config import (
    PORTFOLIOS, N_PORTFOLIOS, VOLL, TOTAL_CAPACITY_MW,
    DBG_N_REPS, FACTORIAL_N_REPS, FACTORIAL_LEVELS,
    KN_ALPHA, FIRM_SUBTYPES,
)
from crn import generate_common_inputs
from dispatch import simulate_replication


# =========================================================================
# B. Deterministic Benchmark Gap
# =========================================================================

def deterministic_evaluation() -> Tuple[int, Dict[int, float]]:
    """
    Evaluate all portfolios under deterministic (mean-input) conditions.

    Sets demand noise = 0, all generators available, renewables at mean CF.
    Returns (best_idx, {idx: Y_det for all portfolios}).
    """
    # Generate inputs with zero noise, no outages, fixed-mean renewables
    no_outage = {name: 0.0 for name in FIRM_SUBTYPES}
    inputs = generate_common_inputs(
        base_seed=0, replication=0,
        sigma_d=0.0,
        for_overrides=no_outage,
        renewable_mode="fixed_mean",
    )

    det_Y = {}
    for i, p in enumerate(PORTFOLIOS):
        _, _, _, y = simulate_replication(p, inputs)
        det_Y[i] = y

    best_det = min(det_Y, key=det_Y.get)
    return best_det, det_Y


def compute_dbg(
    kn_selected: int,
    base_seed: int = 42,
    n_reps: int = DBG_N_REPS,
) -> Dict[str, Any]:
    """
    Compute the Deterministic Benchmark Gap.

    Compares the KN-selected portfolio against the deterministic-best
    using paired CRN replications.

    Returns dict with DBG estimate, CI, and component data.
    """
    det_best, det_Y = deterministic_evaluation()

    if det_best == kn_selected:
        return {
            'det_best': det_best,
            'kn_selected': kn_selected,
            'same_portfolio': True,
            'dbg': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'det_Y': det_Y,
            'message': ('Deterministic and stochastic methods select the same '
                       'portfolio. DBG = 0 by construction.'),
        }

    # Paired evaluation
    Y_det = []
    Y_kn = []
    for rep in range(n_reps):
        inputs = generate_common_inputs(base_seed=base_seed + 10000, replication=rep)
        _, _, _, y_det = simulate_replication(PORTFOLIOS[det_best], inputs)
        _, _, _, y_kn = simulate_replication(PORTFOLIOS[kn_selected], inputs)
        Y_det.append(y_det)
        Y_kn.append(y_kn)

    D = np.array(Y_det) - np.array(Y_kn)
    D_bar = D.mean()
    S_D = D.std(ddof=1)
    t_crit = stats.t.ppf(0.975, n_reps - 1)
    ci_hw = t_crit * S_D / np.sqrt(n_reps)

    return {
        'det_best': det_best,
        'kn_selected': kn_selected,
        'same_portfolio': False,
        'dbg': D_bar,
        'ci_lower': D_bar - ci_hw,
        'ci_upper': D_bar + ci_hw,
        'significant': (D_bar - ci_hw) > 0,  # CI excludes zero and DBG > 0
        'D_bar': D_bar,
        'S_D': S_D,
        'n_reps': n_reps,
        'Y_det_mean': np.mean(Y_det),
        'Y_kn_mean': np.mean(Y_kn),
        'det_Y': det_Y,
    }


# =========================================================================
# C. Sensitivity Analysis (2^3 Factorial on Top-k Survivors)
# =========================================================================

def generate_factorial_configs() -> List[Dict]:
    """
    Generate all 8 factor-level combinations for the 2^3 factorial.

    Factors: (A) demand noise, (B) forced outages, (C) renewable variability.
    """
    configs = []
    levels = FACTORIAL_LEVELS
    for a_level in ['low', 'high']:
        for b_level in ['low', 'high']:
            for c_level in ['low', 'high']:
                configs.append({
                    'label': f"A={a_level[0].upper()},B={b_level[0].upper()},C={c_level[0].upper()}",
                    'a_level': a_level,
                    'b_level': b_level,
                    'c_level': c_level,
                    'sigma_d': levels['demand_noise'][a_level],
                    'for_overrides': levels['forced_outage'][b_level],
                    'renewable_mode': 'fixed_mean' if c_level == 'low' else 'stochastic',
                })
    return configs


def run_factorial(
    survivor_indices: List[int],
    base_seed: int = 42,
    n_reps: int = FACTORIAL_N_REPS,
) -> Dict[str, Any]:
    """
    Run the 2^3 factorial experiment on the specified portfolios.

    Returns dict with:
        'results'  : nested dict [config_label][portfolio_idx] -> list of Y values
        'means'    : nested dict [config_label][portfolio_idx] -> mean Y
        'best_per_config' : dict [config_label] -> best portfolio idx
        'configs'  : list of config dicts
        'main_effects' : dict of factor -> effect size
        'interactions' : dict of factor pair -> interaction size
    """
    configs = generate_factorial_configs()
    results = {}
    means = {}
    best_per_config = {}

    for cfg in configs:
        results[cfg['label']] = {}
        means[cfg['label']] = {}

        for idx in survivor_indices:
            Y_vals = []
            for rep in range(n_reps):
                inputs = generate_common_inputs(
                    base_seed=base_seed + 20000,
                    replication=rep,
                    sigma_d=cfg['sigma_d'],
                    for_overrides=cfg['for_overrides'],
                    renewable_mode=cfg['renewable_mode'],
                )
                _, _, _, y = simulate_replication(PORTFOLIOS[idx], inputs)
                Y_vals.append(y)

            results[cfg['label']][idx] = Y_vals
            means[cfg['label']][idx] = np.mean(Y_vals)

        best_per_config[cfg['label']] = min(
            survivor_indices, key=lambda i: means[cfg['label']][i]
        )

    # Compute main effects and interactions
    # Use the best portfolio's mean Y as the response for each config
    main_effects, interactions = _compute_factorial_effects(configs, means, survivor_indices)

    return {
        'results': results,
        'means': means,
        'best_per_config': best_per_config,
        'configs': configs,
        'main_effects': main_effects,
        'interactions': interactions,
        'survivor_indices': survivor_indices,
    }


def _compute_factorial_effects(
    configs: List[Dict],
    means: Dict,
    survivor_indices: List[int],
) -> Tuple[Dict, Dict]:
    """
    Compute main effects and two-factor interactions.

    The response variable is the performance gap between the best and worst
    survivor portfolio within each factor-level combination.
    """
    # Response: gap between best and worst survivor mean Y
    responses = {}
    for cfg in configs:
        best_mean = min(means[cfg['label']][i] for i in survivor_indices)
        worst_mean = max(means[cfg['label']][i] for i in survivor_indices)
        responses[cfg['label']] = worst_mean - best_mean

    factors = {'A': 'a_level', 'B': 'b_level', 'C': 'c_level'}
    factor_names = {'A': 'Demand noise', 'B': 'Forced outages', 'C': 'Renewable variability'}

    # Main effects: average response at high level minus average at low level
    main_effects = {}
    for factor_key, level_key in factors.items():
        high_vals = [responses[cfg['label']] for cfg in configs
                     if cfg[level_key] == 'high']
        low_vals = [responses[cfg['label']] for cfg in configs
                    if cfg[level_key] == 'low']
        effect = np.mean(high_vals) - np.mean(low_vals)
        main_effects[factor_names[factor_key]] = effect

    # Two-factor interactions
    interactions = {}
    factor_list = list(factors.keys())
    for i, j in combinations(range(3), 2):
        fi, fj = factor_list[i], factor_list[j]
        li, lj = factors[fi], factors[fj]
        name = f"{factor_names[fi]} x {factor_names[fj]}"

        hh = np.mean([responses[c['label']] for c in configs
                       if c[li] == 'high' and c[lj] == 'high'])
        hl = np.mean([responses[c['label']] for c in configs
                       if c[li] == 'high' and c[lj] == 'low'])
        lh = np.mean([responses[c['label']] for c in configs
                       if c[li] == 'low' and c[lj] == 'high'])
        ll = np.mean([responses[c['label']] for c in configs
                       if c[li] == 'low' and c[lj] == 'low'])
        interactions[name] = 0.5 * ((hh - hl) - (lh - ll))

    return main_effects, interactions


# =========================================================================
# E. VOLL Sensitivity (post-processing robustness)
# =========================================================================

def run_voll_sensitivity(
    components: Dict[int, Dict[str, List[float]]],
    portfolio_indices: List[int],
    voll_values: List[float] = None,
) -> Dict[str, Any]:
    """
    Recompute expected objective under alternate VOLL values.

    Uses already-simulated component outputs:
        Y = F + C + lambda * U
    so this check is computationally cheap and preserves CRN pairing.
    """
    if voll_values is None:
        voll_values = [5_000, 10_000, 15_000]

    summary = {}
    for lam in voll_values:
        means = {}
        for idx in portfolio_indices:
            F = np.array(components[idx]['F'])
            C = np.array(components[idx]['C'])
            U = np.array(components[idx]['U'])
            means[idx] = np.mean(F + C + lam * U)

        ranking = sorted(means.keys(), key=lambda i: means[i])
        summary[lam] = {
            'means': means,
            'ranking': ranking,
            'best': ranking[0],
        }

    baseline_best = summary[voll_values[0]]['best']
    stable = all(summary[lam]['best'] == baseline_best for lam in voll_values)
    return {
        'voll_values': voll_values,
        'by_voll': summary,
        'stable_best': stable,
    }


# =========================================================================
# D. Confidence Intervals with Bonferroni Correction
# =========================================================================

def compute_confidence_intervals(
    Y_data: Dict[int, List[float]],
    portfolio_indices: List[int],
    alpha: float = KN_ALPHA,
) -> Dict[str, Any]:
    """
    Compute individual and Bonferroni-corrected pairwise CIs.

    Parameters
    ----------
    Y_data : dict[int, list[float]]
        Raw Y values from KN or independent runs.
    portfolio_indices : list[int]
        Indices to include.
    alpha : float
        Familywise error rate.

    Returns
    -------
    dict with 'individual_cis' and 'pairwise_cis'.
    """
    # Individual CIs (alpha per comparison)
    individual_cis = {}
    for idx in portfolio_indices:
        y = np.array(Y_data[idx])
        n = len(y)
        mean = y.mean()
        se = y.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
        hw = t_crit * se
        rhw = hw / abs(mean) if mean != 0 else np.inf
        individual_cis[idx] = {
            'mean': mean, 'se': se, 'ci_lower': mean - hw,
            'ci_upper': mean + hw, 'half_width': hw,
            'relative_hw': rhw, 'n': n,
        }

    # Pairwise CIs with Bonferroni correction
    pairs = list(combinations(portfolio_indices, 2))
    n_pairs = len(pairs)
    alpha_bonf = alpha / n_pairs if n_pairs > 0 else alpha

    pairwise_cis = {}
    for i, l in pairs:
        yi = np.array(Y_data[i])
        yl = np.array(Y_data[l])
        n = min(len(yi), len(yl))
        D = yi[:n] - yl[:n]
        D_bar = D.mean()
        S_D = D.std(ddof=1)
        t_crit = stats.t.ppf(1 - alpha_bonf / 2, n - 1)
        hw = t_crit * S_D / np.sqrt(n)
        pairwise_cis[(i, l)] = {
            'diff_mean': D_bar, 'se': S_D / np.sqrt(n),
            'ci_lower': D_bar - hw, 'ci_upper': D_bar + hw,
            'significant': (D_bar - hw > 0) or (D_bar + hw < 0),
            'n': n,
        }

    return {
        'individual_cis': individual_cis,
        'pairwise_cis': pairwise_cis,
        'alpha_bonf': alpha_bonf,
        'n_pairs': n_pairs,
    }


def format_ci_results(
    ci_results: Dict, portfolios: list
) -> str:
    """Format CI results as readable text."""
    lines = []
    lines.append("Individual 95% Confidence Intervals:")
    lines.append(f"  {'Portfolio':<22} {'Mean ($M)':>12} {'95% CI':>28} {'RHW':>8}")
    lines.append(f"  {'-'*72}")
    for idx, ci in sorted(ci_results['individual_cis'].items()):
        lines.append(
            f"  {portfolios[idx].label:<22} {ci['mean']/1e6:>12,.1f} "
            f"[{ci['ci_lower']/1e6:>12,.1f}, {ci['ci_upper']/1e6:>12,.1f}] "
            f"{ci['relative_hw']*100:>7.2f}%"
        )

    lines.append(f"\nBonferroni-corrected pairwise comparisons "
                 f"(alpha_pair = {ci_results['alpha_bonf']:.6f}):")
    lines.append(f"  {'Pair':<42} {'Diff ($M)':>12} {'Significant':>12}")
    lines.append(f"  {'-'*68}")
    for (i, l), ci in sorted(ci_results['pairwise_cis'].items()):
        pair_label = f"{portfolios[i].label} - {portfolios[l].label}"
        lines.append(
            f"  {pair_label:<42} {ci['diff_mean']/1e6:>12,.1f} "
            f"{'Yes' if ci['significant'] else 'No':>12}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test of DBG
    print("Deterministic evaluation:")
    det_best, det_Y = deterministic_evaluation()
    print(f"  Deterministic best: {PORTFOLIOS[det_best].label} "
          f"(Y = ${det_Y[det_best]/1e6:,.1f}M)")

    # Show top 5 deterministic
    sorted_det = sorted(det_Y.items(), key=lambda x: x[1])
    print("  Top 5 deterministic:")
    for idx, y in sorted_det[:5]:
        print(f"    {PORTFOLIOS[idx].label}: ${y/1e6:,.1f}M")
