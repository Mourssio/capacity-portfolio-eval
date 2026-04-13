"""
enhancements.py — Additional analyses for report quality.

2. CRN efficiency comparison (CRN vs independent sampling)
3. Cost decomposition stacked bar chart
5. Relative half-width convergence plot
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

from config import (
    PORTFOLIOS, KN_ALPHA, KN_DELTA, KN_N0, KN_MAX_REPS, VOLL,
)
from crn import generate_common_inputs
from dispatch import simulate_replication
from kn import kn_procedure, compute_kn_constants

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


# =========================================================================
# 2. CRN Efficiency Comparison
# =========================================================================

def crn_efficiency_comparison(top_k, n_reps=200, base_seed=42):
    """
    Compare variance of pairwise differences with CRN vs independent sampling.

    With CRN: all portfolios share the same random inputs per replication.
    Without CRN: each portfolio gets independent random inputs.

    Returns dict with efficiency ratios for each pair.
    """
    print("  Running CRN evaluation...")
    Y_crn = {i: [] for i in top_k}
    for rep in range(n_reps):
        inputs = generate_common_inputs(base_seed=base_seed, replication=rep)
        for i in top_k:
            _, _, _, y = simulate_replication(PORTFOLIOS[i], inputs)
            Y_crn[i].append(y)

    print("  Running independent evaluation...")
    Y_ind = {i: [] for i in top_k}
    for i in top_k:
        for rep in range(n_reps):
            # Each portfolio gets its OWN random inputs (different seed offset)
            inputs = generate_common_inputs(
                base_seed=base_seed + 50000 + i * 10000, replication=rep
            )
            _, _, _, y = simulate_replication(PORTFOLIOS[i], inputs)
            Y_ind[i].append(y)

    # Compute pairwise variance of differences
    from itertools import combinations
    results = {}
    for i, l in combinations(top_k, 2):
        D_crn = np.array(Y_crn[i]) - np.array(Y_crn[l])
        D_ind = np.array(Y_ind[i]) - np.array(Y_ind[l])

        var_crn = D_crn.var(ddof=1)
        var_ind = D_ind.var(ddof=1)

        corr_crn = np.corrcoef(Y_crn[i], Y_crn[l])[0, 1]

        results[(i, l)] = {
            'var_crn': var_crn,
            'var_ind': var_ind,
            'efficiency_ratio': var_ind / var_crn if var_crn > 0 else np.inf,
            'correlation': corr_crn,
        }

    return results


def plot_crn_efficiency(results, portfolios, top_k):
    """Bar chart comparing Var(D) with CRN vs independent."""
    from itertools import combinations
    pairs = list(combinations(top_k, 2))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pairs))
    width = 0.35

    var_crn = [results[p]['var_crn'] / 1e12 for p in pairs]
    var_ind = [results[p]['var_ind'] / 1e12 for p in pairs]
    labels = [f"{portfolios[i].label}\nvs\n{portfolios[l].label}" for i, l in pairs]

    bars1 = ax.bar(x - width/2, var_ind, width, label='Independent', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, var_crn, width, label='CRN', color='#2ca02c', alpha=0.8)

    # Annotate efficiency ratios
    for idx, p in enumerate(pairs):
        ratio = results[p]['efficiency_ratio']
        ax.annotate(f'{ratio:.1f}x', xy=(x[idx], max(var_ind[idx], var_crn[idx])),
                    xytext=(0, 8), textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold', color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Var(Y_i - Y_l) (x10^12 $^2)')
    ax.set_title('CRN Variance Reduction: Pairwise Difference Variance')
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig7_crn_efficiency.png'), dpi=150)
    plt.close(fig)


# =========================================================================
# 3. Cost Decomposition Stacked Bar Chart
# =========================================================================

def plot_cost_decomposition(components, portfolios, top_k):
    """Stacked bar chart: F_i (fixed) + C_ij (operating) + lambda*U_ij (penalty)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [portfolios[i].label for i in top_k]
    x = np.arange(len(top_k))

    F = [np.mean(components[i]['F']) / 1e6 for i in top_k]
    C = [np.mean(components[i]['C']) / 1e6 for i in top_k]
    P = [np.mean(components[i]['U']) * VOLL / 1e6 for i in top_k]

    bars_f = ax.bar(x, F, label='Fixed cost $F_i$', color='#1f77b4', edgecolor='black', linewidth=0.3)
    bars_c = ax.bar(x, C, bottom=F, label='Operating cost $C_{ij}$', color='#ff7f0e',
                    edgecolor='black', linewidth=0.3)
    bars_p = ax.bar(x, P, bottom=[f + c for f, c in zip(F, C)],
                    label='Adequacy penalty $\\lambda U_{ij}$', color='#d62728',
                    edgecolor='black', linewidth=0.3)

    # Annotate totals
    for idx in range(len(top_k)):
        total = F[idx] + C[idx] + P[idx]
        ax.text(x[idx], total + max(P) * 0.03, f'${total:,.0f}M',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Annual Cost ($M)')
    ax.set_title('Cost Decomposition: Fixed + Operating + Adequacy Penalty')
    ax.legend(loc='upper left', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}M'))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig8_cost_decomposition.png'), dpi=150)
    plt.close(fig)


# =========================================================================
# 5. Relative Half-Width Convergence Plot
# =========================================================================

def plot_rhw_convergence(Y_data, portfolios, top_k, alpha=0.05):
    """
    Plot relative half-width vs number of replications.
    Shows how precision improves with sample size.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, color in zip(top_k, COLORS):
        y = np.array(Y_data[idx])
        n_max = len(y)
        ns = np.arange(10, n_max + 1)
        rhws = []
        for n in ns:
            sample = y[:n]
            mean = sample.mean()
            se = sample.std(ddof=1) / np.sqrt(n)
            t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
            hw = t_crit * se
            rhw = hw / abs(mean) if mean != 0 else np.inf
            rhws.append(rhw * 100)  # Convert to percentage

        ax.plot(ns, rhws, label=portfolios[idx].label, color=color, linewidth=1.5)

    ax.axhline(5, color='black', linestyle='--', linewidth=0.8, label='5% target')
    ax.axhline(1, color='gray', linestyle=':', linewidth=0.8, label='1% target')

    ax.set_xlabel('Number of Replications')
    ax.set_ylabel('Relative Half-Width (%)')
    ax.set_title('Convergence of Relative Half-Width (95% CI)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(10, max(rhws[:5]) if rhws else 10))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig9_rhw_convergence.png'), dpi=150)
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def run_enhancements(main_results):
    """Run all enhancement analyses using results from main.py."""
    top_k = main_results['top_k']
    components = main_results['extended_eval']['components']
    Y_data = main_results['extended_eval']['Y']

    print("\n" + "=" * 70)
    print("ENHANCEMENT ANALYSES")
    print("=" * 70)

    # 2. CRN efficiency
    print("\n--- CRN Efficiency Comparison ---")
    crn_results = crn_efficiency_comparison(top_k, n_reps=100)
    for (i, l), r in crn_results.items():
        print(f"  {PORTFOLIOS[i].label} vs {PORTFOLIOS[l].label}: "
              f"efficiency = {r['efficiency_ratio']:.1f}x, "
              f"CRN corr = {r['correlation']:.3f}")
    plot_crn_efficiency(crn_results, PORTFOLIOS, top_k)

    # 3. Cost decomposition
    print("\n--- Cost Decomposition ---")
    for i in top_k:
        F = np.mean(components[i]['F']) / 1e6
        C = np.mean(components[i]['C']) / 1e6
        P = np.mean(components[i]['U']) * VOLL / 1e6
        print(f"  {PORTFOLIOS[i].label}: F=${F:,.0f}M + C=${C:,.0f}M + "
              f"lambda*U=${P:,.0f}M = ${F+C+P:,.0f}M")
    plot_cost_decomposition(components, PORTFOLIOS, top_k)

    # 5. RHW convergence
    print("\n--- RHW Convergence ---")
    for i in top_k:
        y = np.array(Y_data[i])
        rhw_final = stats.t.ppf(0.975, len(y)-1) * y.std(ddof=1) / np.sqrt(len(y)) / y.mean() * 100
        print(f"  {PORTFOLIOS[i].label}: RHW at n=200 = {rhw_final:.2f}%")
    plot_rhw_convergence(Y_data, PORTFOLIOS, top_k)

    print("\n  Enhancement figures saved to figures/")
    return {'crn_efficiency': crn_results}


if __name__ == "__main__":
    # Quick standalone test
    print("Run via main.py to use full results.")
