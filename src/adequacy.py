"""
adequacy.py — Stochastic capacity adequacy search.

For each portfolio mix (firm/renewable/storage fractions), finds the minimum
total installed capacity such that expected unserved energy ≤ threshold.
This is a one-dimensional stochastic root-finding problem solved via bisection.

The result: each portfolio's "cost of adequacy" — the minimum total system cost
to achieve reliable supply. The cheapest adequate portfolio may differ from the
cheapest portfolio at a fixed capacity budget.

This bridges portfolio evaluation (ranking & selection) with capacity expansion
planning (Park & Baldick's domain).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import time

from config import (
    PORTFOLIOS, TOTAL_CAPACITY_MW, VOLL,
)
from crn import generate_common_inputs
from dispatch import simulate_replication

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def evaluate_ue(portfolio, n_reps, base_seed=42):
    """
    Run n_reps replications and return mean UE and mean Y.
    """
    ue_vals = []
    y_vals = []
    for rep in range(n_reps):
        inputs = generate_common_inputs(base_seed=base_seed, replication=rep)
        F, C, U, Y = simulate_replication(portfolio, inputs)
        ue_vals.append(U)
        y_vals.append(Y)
    return np.mean(ue_vals), np.mean(y_vals), np.std(y_vals)


def find_adequate_capacity(
    base_portfolio,
    ue_threshold=10.0,       # MWh/year (near-zero on a ~50 TWh system)
    cap_lower=None,
    cap_upper=None,
    cap_tolerance=100,       # MW precision
    n_reps_search=50,        # Replications per bisection step
    n_reps_final=200,        # Replications for final evaluation
    base_seed=42,
):
    """
    Binary search for minimum total capacity achieving mean UE ≤ threshold.

    Returns dict with:
      - adequate_capacity_mw: minimum total capacity for adequacy
      - annualized_cost: total annualized cost at adequate capacity
      - mean_ue: confirmed mean UE at adequate capacity
      - mean_y: mean Y at adequate capacity
      - search_history: list of (capacity, mean_ue) at each bisection step
    """
    if cap_lower is None:
        cap_lower = TOTAL_CAPACITY_MW
    if cap_upper is None:
        cap_upper = TOTAL_CAPACITY_MW * 3  # 3x should be more than enough

    history = []

    # Verify upper bound is adequate
    p_upper = base_portfolio.scaled_to(cap_upper)
    ue_upper, _, _ = evaluate_ue(p_upper, n_reps_search, base_seed)
    history.append((cap_upper, ue_upper))
    if ue_upper > ue_threshold:
        print(f"    WARNING: upper bound {cap_upper:,.0f} MW still inadequate "
              f"(UE={ue_upper:,.0f}). Increase cap_upper.")
        return None

    # Verify lower bound is inadequate (otherwise already adequate)
    p_lower = base_portfolio.scaled_to(cap_lower)
    ue_lower, _, _ = evaluate_ue(p_lower, n_reps_search, base_seed)
    history.append((cap_lower, ue_lower))
    if ue_lower <= ue_threshold:
        # Already adequate at current capacity
        p_final = base_portfolio.scaled_to(cap_lower)
        ue_final, y_final, std_y = evaluate_ue(p_final, n_reps_final, base_seed)
        return {
            'adequate_capacity_mw': cap_lower,
            'annualized_cost': p_final.annualized_fixed_cost(),
            'mean_ue': ue_final,
            'mean_y': y_final,
            'std_y': std_y,
            'search_history': history,
            'steps': 0,
        }

    # Bisection
    steps = 0
    while (cap_upper - cap_lower) > cap_tolerance:
        cap_mid = (cap_lower + cap_upper) / 2
        p_mid = base_portfolio.scaled_to(cap_mid)
        ue_mid, _, _ = evaluate_ue(p_mid, n_reps_search, base_seed)
        history.append((cap_mid, ue_mid))
        steps += 1

        if ue_mid <= ue_threshold:
            cap_upper = cap_mid  # Adequate; try less capacity
        else:
            cap_lower = cap_mid  # Inadequate; need more capacity

    # Final evaluation at the adequate capacity (upper bound)
    adequate_cap = cap_upper
    p_final = base_portfolio.scaled_to(adequate_cap)
    ue_final, y_final, std_y = evaluate_ue(p_final, n_reps_final, base_seed)
    fi = p_final.annualized_fixed_cost()

    return {
        'adequate_capacity_mw': adequate_cap,
        'annualized_cost': fi,
        'mean_ue': ue_final,
        'mean_y': y_final,
        'std_y': std_y,
        'search_history': history,
        'steps': steps,
    }


def plot_adequacy_results(results, portfolios, top_k):
    """
    Two-panel figure:
      Left: minimum adequate capacity per portfolio (bar chart)
      Right: annualized cost at adequate capacity (stacked: fixed + operating)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = [portfolios[i].label for i in top_k]
    x = np.arange(len(top_k))

    # Left panel: adequate capacity
    ax = axes[0]
    caps = [results[i]['adequate_capacity_mw'] / 1000 for i in top_k]
    bars = ax.bar(x, caps, color=COLORS[:len(top_k)], edgecolor='black', linewidth=0.5)
    ax.axhline(TOTAL_CAPACITY_MW / 1000, color='black', linestyle='--', linewidth=1,
               label=f'Fixed budget ({TOTAL_CAPACITY_MW/1000:.0f} GW)')
    for idx, c in enumerate(caps):
        ax.text(x[idx], c + 0.2, f'{c:.1f} GW', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Total Installed Capacity (GW)')
    ax.set_title('Minimum Capacity for Adequacy (EUE ~= 0)')
    ax.legend(fontsize=8)

    # Right panel: annualized cost at adequate capacity
    ax = axes[1]
    costs = [results[i]['annualized_cost'] / 1e6 for i in top_k]
    y_means = [results[i]['mean_y'] / 1e6 for i in top_k]
    bars = ax.bar(x, costs, color=COLORS[:len(top_k)], edgecolor='black', linewidth=0.5,
                  label='Fixed cost $F_i$')

    for idx in range(len(top_k)):
        ax.text(x[idx], costs[idx] + max(costs) * 0.02,
                f'${costs[idx]:,.0f}M', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Annualized Fixed Cost ($M)')
    ax.set_title('Cost of Adequacy: Capital Required for Reliable Supply')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}M'))

    fig.suptitle('Stochastic Adequacy Search: Minimum Capacity for Zero Unserved Energy',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig10_adequacy_search.png'), dpi=150)
    plt.close(fig)


def plot_search_convergence(results, portfolios, top_k):
    """Plot bisection convergence: UE vs capacity for each portfolio."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, color in zip(top_k, COLORS):
        history = results[idx]['search_history']
        caps = [h[0] / 1000 for h in history]
        ues = [max(h[1], 0.1) for h in history]  # Floor at 0.1 for log scale
        # Sort by capacity for clean plotting
        sorted_pairs = sorted(zip(caps, ues))
        caps_s, ues_s = zip(*sorted_pairs)
        ax.plot(caps_s, ues_s, 'o-', color=color, label=portfolios[idx].label,
                markersize=5, linewidth=1.5)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='Threshold (1 MWh)')
    ax.set_yscale('log')
    ax.set_xlabel('Total Installed Capacity (GW)')
    ax.set_ylabel('Mean Unserved Energy (MWh/year)')
    ax.set_title('Bisection Search: UE vs Total Capacity')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig11_search_convergence.png'), dpi=150)
    plt.close(fig)


def run_adequacy_search(top_k, ue_threshold=10.0, n_reps_search=50, n_reps_final=200):
    """Run adequacy search for all top-k portfolios."""
    print("\n" + "=" * 70)
    print("STOCHASTIC ADEQUACY SEARCH")
    print("=" * 70)
    print(f"  Objective: Find minimum total capacity for EUE <= {ue_threshold} MWh/year")
    print(f"  Bisection: {n_reps_search} reps/step, {n_reps_final} reps for final eval")
    print()

    t0 = time.time()
    results = {}

    for i in top_k:
        p = PORTFOLIOS[i]
        print(f"  Searching {p.label}...")
        r = find_adequate_capacity(
            p,
            ue_threshold=ue_threshold,
            n_reps_search=n_reps_search,
            n_reps_final=n_reps_final,
            cap_tolerance=100,
        )
        if r is None:
            print("    FAILED - could not find adequate capacity")
            continue
        results[i] = r
        print(f"    Adequate at {r['adequate_capacity_mw']:,.0f} MW "
              f"({r['steps']} bisection steps)")
        print(f"    Fixed cost: ${r['annualized_cost']/1e6:,.0f}M, "
              f"Mean UE: {r['mean_ue']:,.1f} MWh, "
              f"Mean Y: ${r['mean_y']/1e6:,.0f}M")

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")

    # Summary table
    print(f"\n  {'Portfolio':<22} {'Adequate Cap':>14} {'Fixed Cost':>12} "
          f"{'Mean Y':>10} {'Mean UE':>10}")
    print(f"  {'-'*70}")
    for i in top_k:
        if i not in results:
            continue
        r = results[i]
        print(f"  {PORTFOLIOS[i].label:<22} {r['adequate_capacity_mw']:>11,.0f} MW "
              f"${r['annualized_cost']/1e6:>10,.0f}M "
              f"${r['mean_y']/1e6:>8,.0f}M "
              f"{r['mean_ue']:>9,.1f}")

    # Find cheapest adequate portfolio
    if results:
        cheapest = min(results, key=lambda i: results[i]['mean_y'])
        print(f"\n  Cheapest adequate portfolio: {PORTFOLIOS[cheapest].label} "
              f"(${results[cheapest]['mean_y']/1e6:,.0f}M)")

    # Plots
    plot_adequacy_results(results, PORTFOLIOS, top_k)
    plot_search_convergence(results, PORTFOLIOS, top_k)
    print("  Figures saved: fig10_adequacy_search.png, fig11_search_convergence.png")

    return results


if __name__ == "__main__":
    # Standalone test with top-4
    top_k = [4, 10, 3, 9]
    run_adequacy_search(top_k)
