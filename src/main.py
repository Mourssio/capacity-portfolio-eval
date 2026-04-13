"""
main.py — Orchestrates all experiments and produces outputs.

Run: python src/main.py
"""

import time
import json
import numpy as np
from config import PORTFOLIOS, N_PORTFOLIOS, KN_ALPHA, KN_DELTA, KN_N0
from crn import generate_common_inputs
from dispatch import simulate_replication
from kn import kn_procedure, format_kn_results, compute_kn_constants
from experiment import (
    compute_dbg, run_factorial, compute_confidence_intervals,
    format_ci_results, deterministic_evaluation, run_voll_sensitivity,
)
from plots import generate_all_plots
from enhancements import run_enhancements
from adequacy import run_adequacy_search
from ucap_comparison import run_ucap_comparison


def sim_fn(i, inputs):
    """Wrapper for KN: returns Y_ij only."""
    return simulate_replication(PORTFOLIOS[i], inputs)[3]


def run_all():
    """Execute the full experimental pipeline."""
    t_start = time.time()
    results = {}

    # ==================================================================
    # Stage 1: Kim-Nelson Selection (k=16)
    # ==================================================================
    print("=" * 70)
    print("STAGE 1: Kim-Nelson Sequential Elimination")
    print("=" * 70)
    t1 = time.time()

    kn_results = kn_procedure(sim_fn, generate_common_inputs)
    results['kn'] = kn_results

    print(format_kn_results(kn_results, PORTFOLIOS))
    print(f"\n  Time: {time.time() - t1:.1f}s")

    # Identify top-k survivors for factorial
    # Use top 4 by sample mean from KN data if more than 4 survive
    all_means_at_kn_end = {
        i: np.mean(kn_results['all_data'][i][:kn_results['total_reps']])
        for i in range(N_PORTFOLIOS)
        if len(kn_results['all_data'][i]) >= kn_results['total_reps']
    }
    # For portfolios eliminated early, use available data
    for i in range(N_PORTFOLIOS):
        if i not in all_means_at_kn_end:
            all_means_at_kn_end[i] = np.mean(kn_results['all_data'][i])

    sorted_by_mean = sorted(all_means_at_kn_end.items(), key=lambda x: x[1])
    top_k = [idx for idx, _ in sorted_by_mean[:4]]
    results['top_k'] = top_k

    print(f"\n  Top-4 portfolios for factorial analysis:")
    for idx in top_k:
        print(f"    {PORTFOLIOS[idx].label}: ${all_means_at_kn_end[idx]/1e6:,.1f}M")

    # ==================================================================
    # Stage 2: Extended CRN Evaluation (top-k, 200 replications)
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("STAGE 2: Extended CRN Evaluation (Top-4)")
    print("=" * 70)
    t2 = time.time()

    N_EVAL = 200
    Y_extended = {i: [] for i in top_k}
    # Also collect component data for reporting
    components = {i: {'F': [], 'C': [], 'U': []} for i in top_k}

    for rep in range(N_EVAL):
        inputs = generate_common_inputs(base_seed=42, replication=rep)
        for i in top_k:
            F_i, C_ij, U_ij, Y_ij = simulate_replication(PORTFOLIOS[i], inputs)
            Y_extended[i].append(Y_ij)
            components[i]['F'].append(F_i)
            components[i]['C'].append(C_ij)
            components[i]['U'].append(U_ij)

    results['extended_eval'] = {
        'Y': Y_extended,
        'components': components,
        'n_reps': N_EVAL,
    }

    print(f"  Completed {N_EVAL} replications for top-4 portfolios.")
    print(f"\n  {'Portfolio':<22} {'Mean Y ($M)':>12} {'Mean C ($M)':>12} "
          f"{'Mean UE (MWh)':>14} {'Std Y ($M)':>12}")
    print(f"  {'-'*76}")
    for i in top_k:
        my = np.mean(Y_extended[i])
        mc = np.mean(components[i]['C'])
        mu = np.mean(components[i]['U'])
        sy = np.std(Y_extended[i], ddof=1)
        print(f"  {PORTFOLIOS[i].label:<22} {my/1e6:>12,.1f} {mc/1e6:>12,.1f} "
              f"{mu:>14,.0f} {sy/1e6:>12,.1f}")

    print(f"\n  Time: {time.time() - t2:.1f}s")

    # ==================================================================
    # Stage 3: Confidence Intervals
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("STAGE 3: Confidence Intervals (Bonferroni-corrected)")
    print("=" * 70)

    ci_results = compute_confidence_intervals(Y_extended, top_k)
    results['ci'] = ci_results
    print(format_ci_results(ci_results, PORTFOLIOS))

    # ==================================================================
    # Stage 4: Deterministic Benchmark Gap
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("STAGE 4: Deterministic Benchmark Gap")
    print("=" * 70)
    t4 = time.time()

    dbg_results = compute_dbg(kn_results['selected'])
    results['dbg'] = dbg_results

    print(f"  Deterministic best:  {PORTFOLIOS[dbg_results['det_best']].label}")
    print(f"  Stochastic (KN) best: {PORTFOLIOS[dbg_results['kn_selected']].label}")

    if dbg_results['same_portfolio']:
        print(f"  {dbg_results['message']}")
    else:
        print(f"  DBG = ${dbg_results['dbg']/1e6:,.1f}M")
        print(f"  95% CI: [${dbg_results['ci_lower']/1e6:,.1f}M, "
              f"${dbg_results['ci_upper']/1e6:,.1f}M]")
        print(f"  Significant (CI excludes 0): {dbg_results['significant']}")
        print(f"  Interpretation: {'Stochastic selection outperforms' if dbg_results['significant'] else 'No significant difference'}")

    print(f"\n  Time: {time.time() - t4:.1f}s")

    # ==================================================================
    # Stage 5: Factorial Sensitivity Analysis (Top-4)
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("STAGE 5: 2^3 Factorial Sensitivity Analysis")
    print("=" * 70)
    t5 = time.time()

    factorial_results = run_factorial(top_k)
    results['factorial'] = factorial_results

    print(f"\n  Best portfolio per factor-level combination:")
    print(f"  {'Config':<20} {'Best Portfolio':<22} {'Mean Y ($M)':>12}")
    print(f"  {'-'*56}")
    for cfg in factorial_results['configs']:
        best_idx = factorial_results['best_per_config'][cfg['label']]
        best_mean = factorial_results['means'][cfg['label']][best_idx]
        print(f"  {cfg['label']:<20} {PORTFOLIOS[best_idx].label:<22} "
              f"{best_mean/1e6:>12,.1f}")

    print(f"\n  Main effects (on best-worst gap):")
    for factor, effect in sorted(factorial_results['main_effects'].items(),
                                  key=lambda x: abs(x[1]), reverse=True):
        print(f"    {factor:<30} ${effect/1e6:>12,.1f}M")

    print(f"\n  Two-factor interactions:")
    for pair, effect in sorted(factorial_results['interactions'].items(),
                                key=lambda x: abs(x[1]), reverse=True):
        print(f"    {pair:<40} ${effect/1e6:>12,.1f}M")

    # Check if best portfolio changes across configs
    unique_best = set(factorial_results['best_per_config'].values())
    print(f"\n  Ranking stability: {'Best portfolio changes across configs'if len(unique_best) > 1 else 'Same best portfolio in all configs'}")
    if len(unique_best) > 1:
        print(f"  Unique best portfolios: {[PORTFOLIOS[i].label for i in unique_best]}")

    print(f"\n  Time: {time.time() - t5:.1f}s")

    # ==================================================================
    # Stage 6: VOLL Sensitivity (post-processing robustness)
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("STAGE 6: VOLL Sensitivity")
    print("=" * 70)
    voll_results = run_voll_sensitivity(components, top_k, [5_000, 10_000, 15_000])
    results['voll_sensitivity'] = voll_results
    print(f"  {'VOLL ($/MWh)':<14} {'Best Portfolio':<22} {'Mean Y ($M)':>12}")
    print(f"  {'-'*52}")
    for lam in voll_results['voll_values']:
        best_idx = voll_results['by_voll'][lam]['best']
        best_mean = voll_results['by_voll'][lam]['means'][best_idx]
        print(f"  {lam:<14,.0f} {PORTFOLIOS[best_idx].label:<22} {best_mean/1e6:>12,.1f}")
    print(f"\n  Stable best across tested VOLL range: {voll_results['stable_best']}")

    # ==================================================================
    # Stage 7: Generate Figures
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("STAGE 7: Generating Figures")
    print("=" * 70)

    generate_all_plots(results, PORTFOLIOS)
    print("  Figures saved to figures/")

    # ==================================================================
    # Stage 8: Enhancement Analyses
    # ==================================================================
    enhancement_results = run_enhancements(results)
    results['enhancements'] = enhancement_results

    # ==================================================================
    # Stage 9: Stochastic Adequacy Search
    # ==================================================================
    adequacy_results = run_adequacy_search(top_k)
    results['adequacy'] = adequacy_results

    # ==================================================================
    # Stage 10: UCAP-Normalized Comparison
    # ==================================================================
    # Reshape extended eval data for UCAP comparison
    equal_mw_data = {}
    for i in top_k:
        equal_mw_data[i] = {
            'Y': np.array(results['extended_eval']['Y'][i]),
            'F': np.array(results['extended_eval']['components'][i]['F']),
            'C': np.array(results['extended_eval']['components'][i]['C']),
            'U': np.array(results['extended_eval']['components'][i]['U']),
        }
    ucap_results = run_ucap_comparison(top_k, equal_mw_data)
    results['ucap'] = ucap_results

    # ==================================================================
    # Summary
    # ==================================================================
    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE - Total runtime: {total_time:.1f}s")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  1. KN selects {PORTFOLIOS[kn_results['selected']].label} "
          f"after {kn_results['total_reps']} replications")
    print(f"  2. Deterministic analysis selects "
          f"{PORTFOLIOS[dbg_results['det_best']].label}")
    if not dbg_results['same_portfolio']:
        print(f"  3. DBG = ${dbg_results['dbg']/1e6:,.1f}M - uncertainty "
              f"{'changes' if dbg_results['significant'] else 'does not change'} "
              f"the preferred portfolio")
    print(f"  4. Most impactful factor: "
          f"{max(factorial_results['main_effects'], key=lambda k: abs(factorial_results['main_effects'][k]))}")

    return results


if __name__ == "__main__":
    results = run_all()
