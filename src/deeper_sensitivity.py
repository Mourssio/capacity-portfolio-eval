"""
deeper_sensitivity.py -- Two probes answering structural critiques of the
main result.

(1) Boundary probe. P04 (70/20/10) sits exactly at the upper bound of the
    feasible firm share in the main 10%-step simplex. If the true optimum
    lies outside this region, P04 is only the best within an artificially
    truncated search space. This probe constructs three new portfolios
    that push firm share above 70% and re-ranks them against P04.

(2) Adequate-capacity probe. The main experiment fixes total capacity at
    12 GW, where the adequacy search later shows that even P04 carries
    non-trivial EUE and would need roughly 15.8 GW for full adequacy. The
    critic's concern is that the ranking is happening inside a materially
    inadequate system. This probe re-evaluates the top-4 portfolios at
    three total-capacity levels: 12 GW (main), 15.8 GW (P04 adequate),
    and 18 GW (all top-4 adequate).
"""

import numpy as np
from config import PORTFOLIOS, Portfolio, TOTAL_CAPACITY_MW
from crn import generate_common_inputs
from dispatch import simulate_replication


N_REPS_BOUNDARY = 100
N_REPS_CAPACITY = 100


def make_portfolio(pid, firm, renew, storage, total_mw=TOTAL_CAPACITY_MW):
    assert abs(firm + renew + storage - 1.0) < 1e-9, \
        f"Fractions must sum to 1: {firm}+{renew}+{storage}"
    return Portfolio(
        pid=pid,
        firm_frac=firm,
        renew_frac=renew,
        storage_frac=storage,
        total_capacity_mw=total_mw,
    )


def evaluate_portfolio(portfolio, n_reps, base_seed=42):
    """Run n_reps CRN replications and return (mean_Y, mean_U, std_Y)."""
    Y = []
    U = []
    for rep in range(n_reps):
        inputs = generate_common_inputs(base_seed, rep)
        _, _, U_ij, Y_ij = simulate_replication(portfolio, inputs)
        Y.append(Y_ij)
        U.append(U_ij)
    return float(np.mean(Y)), float(np.mean(U)), float(np.std(Y, ddof=1))


def boundary_probe():
    """
    Compare P04 against three higher-firm portfolios that lie outside
    the original [20%, 70%] firm range.
    """
    print("=" * 70)
    print("BOUNDARY PROBE: is P04 (70% firm) the true optimum?")
    print("=" * 70)
    print(f"  Total capacity fixed at {TOTAL_CAPACITY_MW:,} MW (12 GW)")
    print(f"  {N_REPS_BOUNDARY} CRN replications per portfolio")
    print()

    # P04 from main design (baseline)
    p04 = next(p for p in PORTFOLIOS if p.pid == 4)
    # Three extensions beyond the upper bound
    candidates = [
        ("P04  (70/20/10)   [main]",  p04),
        ("P_b1 (75/15/10)",           make_portfolio(1001, 0.75, 0.15, 0.10)),
        ("P_b2 (80/10/10)",           make_portfolio(1002, 0.80, 0.10, 0.10)),
        ("P_b3 (85/10/05)",           make_portfolio(1003, 0.85, 0.10, 0.05)),
        ("P_b4 (80/15/05)",           make_portfolio(1004, 0.80, 0.15, 0.05)),
    ]

    print(f"  {'Portfolio':<28} {'Mean Y ($M)':>13} {'Mean EUE (MWh)':>15} "
          f"{'Std Y ($M)':>12}")
    print(f"  {'-'*70}")
    results = []
    for label, p in candidates:
        my, mu, sy = evaluate_portfolio(p, N_REPS_BOUNDARY)
        results.append((label, my, mu, sy, p))
        print(f"  {label:<28} {my/1e6:>13,.1f} {mu:>15,.0f} {sy/1e6:>12,.1f}")

    # Identify winner
    winner = min(results, key=lambda r: r[1])
    print()
    print(f"  Winner (lowest Mean Y): {winner[0]}")
    p04_my = results[0][1]
    gap = p04_my - winner[1]
    if winner[0].startswith("P04"):
        print(f"  P04 remains best. Boundary probe does NOT alter the main result.")
    else:
        print(f"  New winner beats P04 by ${gap/1e6:,.1f}M. "
              f"Main result was artifacted by the simplex upper bound.")
    return results


def capacity_sweep():
    """
    Re-evaluate the top-4 portfolios from the main run at three total
    capacity levels: 12 GW (main), 15.8 GW, and 18 GW.
    """
    print()
    print("=" * 70)
    print("ADEQUATE-CAPACITY SWEEP: is P04 still best when EUE is small?")
    print("=" * 70)
    print(f"  {N_REPS_CAPACITY} CRN replications per portfolio per capacity")
    print()

    # Fractions for P04, P10, P03, P09 from the main run
    top4_fractions = [
        ("P04 (70/20/10)", 0.70, 0.20, 0.10),
        ("P10 (70/10/20)", 0.70, 0.10, 0.20),
        ("P03 (60/30/10)", 0.60, 0.30, 0.10),
        ("P09 (60/20/20)", 0.60, 0.20, 0.20),
    ]
    capacities = [12_000, 15_800, 18_000]  # MW

    for C in capacities:
        print(f"  Total capacity = {C/1e3:.1f} GW")
        print(f"  {'Portfolio':<20} {'Mean Y ($M)':>13} {'Mean EUE (MWh)':>15} "
              f"{'Std Y ($M)':>12}")
        print(f"  {'-'*60}")
        cap_results = []
        for label, f, r, s in top4_fractions:
            p = make_portfolio(2000, f, r, s, total_mw=C)
            my, mu, sy = evaluate_portfolio(p, N_REPS_CAPACITY)
            cap_results.append((label, my, mu, sy))
            print(f"  {label:<20} {my/1e6:>13,.1f} {mu:>15,.0f} {sy/1e6:>12,.1f}")
        winner = min(cap_results, key=lambda r: r[1])
        print(f"  Winner at {C/1e3:.1f} GW: {winner[0]}")
        print()


def dbg_at_adequate_capacity(total_mw=15_800):
    """
    At adequate capacity, does the DBG between the deterministic winner
    and the stochastic winner remain the same ~$4B, or does it shrink?

    The hypothesis is that DBG is a feature of the inadequate regime and
    collapses when the system is large enough that EUE is near zero.
    """
    print()
    print("=" * 70)
    print(f"DBG AT ADEQUATE CAPACITY ({total_mw/1e3:.1f} GW)")
    print("=" * 70)

    top4 = [
        ("P04 (70/20/10)", 0.70, 0.20, 0.10),
        ("P10 (70/10/20)", 0.70, 0.10, 0.20),
        ("P03 (60/30/10)", 0.60, 0.30, 0.10),
        ("P09 (60/20/20)", 0.60, 0.20, 0.20),
    ]

    # Deterministic: zero noise, zero FOR, fixed-mean renewables, 1 rep
    zero_for = {"nuclear": 0, "flex_firm": 0, "mid_merit": 0, "peaker": 0}
    det_inputs = generate_common_inputs(
        base_seed=42, replication=0,
        sigma_d=0.0, for_overrides=zero_for, renewable_mode="fixed_mean",
    )

    print("  Deterministic (zero noise, zero FOR, fixed-mean renewables):")
    print(f"  {'Portfolio':<20} {'Y ($M)':>12}")
    print(f"  {'-'*36}")
    det_results = []
    for label, f, r, s in top4:
        p = make_portfolio(3000, f, r, s, total_mw=total_mw)
        _, _, _, Y_det = simulate_replication(p, det_inputs)
        det_results.append((label, Y_det))
        print(f"  {label:<20} {Y_det/1e6:>12,.1f}")
    det_winner = min(det_results, key=lambda x: x[1])
    print(f"  Deterministic winner: {det_winner[0]}")

    # Stochastic: 100 CRN reps
    print()
    print("  Stochastic (100 CRN replications):")
    print(f"  {'Portfolio':<20} {'Mean Y ($M)':>13}")
    print(f"  {'-'*36}")
    stoch_results = []
    for label, f, r, s in top4:
        p = make_portfolio(3000, f, r, s, total_mw=total_mw)
        my, _, _ = evaluate_portfolio(p, n_reps=100)
        stoch_results.append((label, my))
        print(f"  {label:<20} {my/1e6:>13,.1f}")
    stoch_winner = min(stoch_results, key=lambda x: x[1])
    print(f"  Stochastic winner: {stoch_winner[0]}")

    # DBG
    det_best_Y_stoch = next(
        s[1] for s in stoch_results if s[0] == det_winner[0]
    )
    dbg = det_best_Y_stoch - stoch_winner[1]
    print()
    print(f"  DBG at {total_mw/1e3:.1f} GW = "
          f"Y_stoch({det_winner[0]}) - Y_stoch({stoch_winner[0]}) "
          f"= ${dbg/1e6:,.1f}M")
    if abs(dbg) < 50:
        print(f"  DBG is within indifference zone ($50M); "
              f"deterministic and stochastic recommendations agree.")
    return {
        "capacity_gw": total_mw / 1e3,
        "det_winner": det_winner[0],
        "stoch_winner": stoch_winner[0],
        "dbg_million": dbg / 1e6,
    }


if __name__ == "__main__":
    boundary_probe()
    capacity_sweep()
    dbg_at_adequate_capacity(total_mw=15_800)
    dbg_at_adequate_capacity(total_mw=18_000)
