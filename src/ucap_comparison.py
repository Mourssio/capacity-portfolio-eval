"""
ucap_comparison.py — UCAP-normalized portfolio comparison.

Scales each portfolio so that all have equal effective capacity (UCAP),
then evaluates under uncertainty. This addresses the critique that
equal-installed-MW comparisons are biased toward firm-heavy portfolios
because they have more UCAP per installed MW.

Under UCAP normalization, renewable-heavy portfolios get more installed
MW (to compensate for low ELCC), which increases their capital cost but
also increases their renewable energy output. The tradeoff between
capital cost and operating cost becomes the central comparison.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import time

from config import (
    PORTFOLIOS, PEAK_DEMAND_MW, TOTAL_CAPACITY_MW, VOLL, ELCC,
)
from crn import generate_common_inputs
from dispatch import simulate_replication

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def evaluate_portfolio(portfolio, n_reps, base_seed=42):
    """Run n_reps replications, return arrays of F, C, U, Y and components."""
    F_vals, C_vals, U_vals, Y_vals = [], [], [], []
    for rep in range(n_reps):
        inputs = generate_common_inputs(base_seed=base_seed, replication=rep)
        F, C, U, Y = simulate_replication(portfolio, inputs)
        F_vals.append(F)
        C_vals.append(C)
        U_vals.append(U)
        Y_vals.append(Y)
    return {
        'F': np.array(F_vals),
        'C': np.array(C_vals),
        'U': np.array(U_vals),
        'Y': np.array(Y_vals),
    }


def plot_ucap_comparison(equal_mw, ucap_norm, portfolios, top_k, target_ucap, ucap_set):
    """
    Two-panel figure:
      Left: UCAP-normalized cost bar chart (all portfolios in ucap_set)
      Right: Cost decomposition stacked bar under UCAP normalization
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    n_colors = max(len(ucap_set), 8)
    cmap = plt.cm.get_cmap('tab10', n_colors)
    colors = [cmap(i) for i in range(len(ucap_set))]

    labels = [portfolios[i].label for i in ucap_set]
    x = np.arange(len(ucap_set))

    # Panel 1: Mean Y under UCAP normalization (sorted by Y)
    ax = axes[0]
    sorted_set = sorted(ucap_set, key=lambda i: ucap_norm[i]['Y'].mean())
    y_vals = [ucap_norm[i]['Y'].mean() / 1e6 for i in sorted_set]
    sorted_labels = [portfolios[i].label for i in sorted_set]
    mw_vals = [ucap_norm[i]['total_mw'] / 1000 for i in sorted_set]
    bar_colors = [cmap(ucap_set.index(i)) for i in sorted_set]

    bars = ax.bar(range(len(sorted_set)), y_vals, color=bar_colors,
                  edgecolor='black', linewidth=0.5)
    for idx in range(len(sorted_set)):
        ax.text(idx, y_vals[idx] + max(y_vals)*0.02,
                f'${y_vals[idx]:,.0f}\n({mw_vals[idx]:.1f} GW)',
                ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(range(len(sorted_set)))
    ax.set_xticklabels(sorted_labels, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Mean Total Cost Y ($M)')
    ax.set_title(f'UCAP-Normalized Cost (Target = {target_ucap/1000:.0f} GW UCAP)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

    # Panel 2: Stacked cost decomposition (sorted same way)
    ax = axes[1]
    F_vals = [ucap_norm[i]['F'].mean() / 1e6 for i in sorted_set]
    C_vals = [ucap_norm[i]['C'].mean() / 1e6 for i in sorted_set]
    P_vals = [ucap_norm[i]['U'].mean() * VOLL / 1e6 for i in sorted_set]

    ax.bar(range(len(sorted_set)), F_vals, label='Fixed $F_i$',
           color='#1f77b4', edgecolor='black', linewidth=0.3)
    ax.bar(range(len(sorted_set)), C_vals, bottom=F_vals,
           label='Operating $C_{ij}$', color='#ff7f0e',
           edgecolor='black', linewidth=0.3)
    ax.bar(range(len(sorted_set)), P_vals,
           bottom=[f+c for f,c in zip(F_vals, C_vals)],
           label='Penalty $\\lambda U_{ij}$', color='#d62728',
           edgecolor='black', linewidth=0.3)

    ax.set_xticks(range(len(sorted_set)))
    ax.set_xticklabels(sorted_labels, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Annual Cost ($M)')
    ax.set_title('UCAP-Normalized Cost Decomposition')
    ax.legend(fontsize=8, loc='upper left')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))

    fig.suptitle('UCAP-Normalized Comparison: Equal Effective Capacity Across Mixes',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig12_ucap_comparison.png'), dpi=150)
    plt.close(fig)


def run_ucap_comparison(top_k, equal_mw_results, n_reps=100, base_seed=42):
    """
    Run UCAP-normalized evaluation.

    Uses an expanded portfolio set to span the full firm-fraction range,
    since the manager's point is that equal-installed-MW comparisons
    pre-select for high-firm portfolios.
    """
    target_ucap = PEAK_DEMAND_MW  # 8,000 MW

    # Expand to 8 portfolios spanning the firm-fraction range
    # Include top-4 plus representative low-firm portfolios
    all_indices = list(range(len(PORTFOLIOS)))
    # Select one portfolio per firm-fraction level
    ucap_set = []
    seen_firm = set()
    # First add top-4
    for i in top_k:
        ff = int(PORTFOLIOS[i].firm_frac * 100)
        if ff not in seen_firm:
            ucap_set.append(i)
            seen_firm.add(ff)
    # Then fill in missing firm fractions, preferring lowest storage
    for i in all_indices:
        ff = int(PORTFOLIOS[i].firm_frac * 100)
        if ff not in seen_firm:
            ucap_set.append(i)
            seen_firm.add(ff)
    ucap_set.sort(key=lambda i: PORTFOLIOS[i].firm_frac, reverse=True)

    print("\n" + "=" * 70)
    print("UCAP-NORMALIZED PORTFOLIO COMPARISON")
    print("=" * 70)
    print(f"  Target UCAP: {target_ucap:,} MW (= peak demand)")
    print(f"  ELCC values: firm ~{ELCC['nuclear']:.0%}-{ELCC['mid_merit']:.0%}, "
          f"wind {ELCC['wind']:.0%}, solar {ELCC['solar']:.0%}, "
          f"storage {ELCC['storage']:.0%}")
    print(f"  Evaluating {len(ucap_set)} portfolios spanning firm "
          f"{min(PORTFOLIOS[i].firm_frac for i in ucap_set):.0%}"
          f"-{max(PORTFOLIOS[i].firm_frac for i in ucap_set):.0%}")
    print()

    # Show sizing table
    print(f"  {'Portfolio':<22} {'ELCC':>6} {'UCAP@12GW':>10} {'Req MW':>10} {'Ratio':>7}")
    print(f"  {'-'*58}")
    for i in ucap_set:
        p = PORTFOLIOS[i]
        elcc = p.weighted_elcc()
        ucap = p.ucap_mw()
        req = target_ucap / elcc
        print(f"  {p.label:<22} {elcc:>6.3f} {ucap:>9,.0f} {req:>9,.0f} {req/TOTAL_CAPACITY_MW:>6.2f}x")

    # Evaluate each portfolio at UCAP-normalized capacity
    print(f"\n  Running {n_reps} replications per UCAP-normalized portfolio...")
    t0 = time.time()
    ucap_results = {}
    for i in ucap_set:
        p_scaled = PORTFOLIOS[i].scaled_to_ucap(target_ucap)
        result = evaluate_portfolio(p_scaled, n_reps, base_seed)
        result['total_mw'] = p_scaled._total_cap
        result['elcc'] = PORTFOLIOS[i].weighted_elcc()
        ucap_results[i] = result

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")

    # Results table
    print(f"\n  {'Portfolio':<22} {'Total MW':>10} {'Mean Y ($M)':>12} "
          f"{'F ($M)':>10} {'C ($M)':>10} {'lambda*U ($M)':>13} {'UE (MWh)':>10}")
    print(f"  {'-'*88}")
    for i in ucap_set:
        r = ucap_results[i]
        F = r['F'].mean() / 1e6
        C = r['C'].mean() / 1e6
        U = r['U'].mean()
        Y = r['Y'].mean() / 1e6
        print(f"  {PORTFOLIOS[i].label:<22} {r['total_mw']:>9,.0f} "
              f"${Y:>10,.0f} ${F:>8,.0f} ${C:>8,.0f} "
              f"${U*VOLL/1e6:>8,.0f} {U:>9,.0f}")

    # Identify best
    best_ucap = min(ucap_set, key=lambda i: ucap_results[i]['Y'].mean())
    best_equal = min(top_k, key=lambda i: equal_mw_results[i]['Y'].mean())

    print(f"\n  Best under equal installed MW (top-4): {PORTFOLIOS[best_equal].label} "
          f"(${equal_mw_results[best_equal]['Y'].mean()/1e6:,.0f}M)")
    print(f"  Best under UCAP normalization (all):   {PORTFOLIOS[best_ucap].label} "
          f"(${ucap_results[best_ucap]['Y'].mean()/1e6:,.0f}M)")

    if best_equal != best_ucap:
        print(f"  >> Ranking changes under UCAP normalization!")
    else:
        print(f"  >> Same portfolio preferred under both comparisons.")

    # Plot
    plot_ucap_comparison(equal_mw_results, ucap_results, PORTFOLIOS,
                         top_k, target_ucap, ucap_set)
    print(f"  Figure saved: fig12_ucap_comparison.png")

    return ucap_results


if __name__ == "__main__":
    top_k = [4, 10, 3, 9]
    # Quick standalone: need equal_mw_results, so just run fresh
    print("Run via main.py for full comparison.")
