"""
ocba.py -- Optimal Computing Budget Allocation (Chen et al. 2000) for
simulation-based ranking and selection, with a head-to-head comparison
against the Kim-Nelson procedure on the 16-portfolio problem.

OCBA is a non-sequential, non-elimination R&S method that allocates a
fixed simulation budget to maximize the probability of correct selection,
concentrating samples on systems most likely to mask the true best.
Classical OCBA assumes independent samples per system (no CRN).

The comparison answers: given the same simulation budget that KN used to
terminate, does OCBA achieve the same selection? How does OCBA distribute
its samples across systems?

References:
  Chen, Lin, Yucesan, Chick (2000). "Simulation Budget Allocation for
  Further Enhancing the Efficiency of Ordinal Optimization." Journal of
  Discrete Event Dynamic Systems 10(3), 251-270.
"""

import time
import numpy as np
from typing import Callable, Dict, Any
from config import PORTFOLIOS, N_PORTFOLIOS, KN_ALPHA, KN_DELTA, KN_N0
from crn import generate_common_inputs
from dispatch import simulate_replication
from kn import kn_procedure


def sim_fn_Y(i, inputs):
    return simulate_replication(PORTFOLIOS[i], inputs)[3]


def ocba_allocation(means: np.ndarray, variances: np.ndarray,
                    target_total: int) -> np.ndarray:
    """
    Compute the OCBA target sample allocation N_1, ..., N_k that
    approximately maximizes P(correct selection) for a total budget
    target_total, given current sample means and variances.

    For minimization:
      N_i proportional to sigma_i^2 / (mu_i - mu_b)^2  for i != b
      N_b = sigma_b * sqrt(sum_{i != b} N_i^2 / sigma_i^2)
    where b = argmin(mu).
    """
    k = len(means)
    b = int(np.argmin(means))
    delta = means - means[b]  # delta[b] = 0, others > 0 (approx)

    ratios = np.zeros(k)
    for i in range(k):
        if i == b:
            continue
        var_i = max(variances[i], 1.0)
        d = delta[i]
        if d < 1e-6:
            # Tie with current best: allocate like the best
            ratios[i] = var_i
        else:
            ratios[i] = var_i / (d * d)

    # N_b via the OCBA formula
    sum_term = 0.0
    for i in range(k):
        if i != b:
            var_i = max(variances[i], 1.0)
            sum_term += (ratios[i] ** 2) / var_i
    ratios[b] = np.sqrt(max(variances[b], 1.0)) * np.sqrt(max(sum_term, 1e-18))

    # Normalize so sum == target_total
    s = ratios.sum()
    if s <= 0:
        return np.full(k, target_total // k, dtype=int)
    N = ratios / s * target_total
    return np.maximum(N.round().astype(int), 1)


def run_ocba(budget: int, k: int = N_PORTFOLIOS, n0: int = KN_N0,
             base_seed: int = 42, reallocation_step: int = None,
             verbose: bool = False) -> Dict[str, Any]:
    """
    Run OCBA on the 16-portfolio problem to a fixed simulation budget.

    Each simulation call uses a fresh (base_seed, global_rep_counter) input
    set, so samples are effectively independent across systems (classical
    OCBA assumption; no pairwise CRN variance reduction). This is a
    deliberately different sampling model from KN-with-CRN and part of
    the comparison.
    """
    if reallocation_step is None:
        reallocation_step = k  # Reallocate after every k new samples

    samples = {i: [] for i in range(k)}
    # Phase 1: uniform n0 per system
    global_rep = 0
    for _ in range(n0):
        inputs = generate_common_inputs(base_seed, global_rep)
        for i in range(k):
            samples[i].append(sim_fn_Y(i, inputs))
        global_rep += 1
    used = n0 * k

    # Phase 2: iterative OCBA reallocation
    iteration = 0
    while used < budget:
        means = np.array([np.mean(samples[i]) for i in range(k)])
        variances = np.array([
            np.var(samples[i], ddof=1) if len(samples[i]) > 1 else 1.0
            for i in range(k)
        ])

        # Target allocation for the *next* step's total budget
        next_total = min(used + reallocation_step, budget)
        target = ocba_allocation(means, variances, next_total)
        current = np.array([len(samples[i]) for i in range(k)])
        deficit = np.maximum(target - current, 0)

        # Cap additions to remaining budget
        remaining = budget - used
        if deficit.sum() > remaining:
            # Distribute remaining among the systems with highest deficit
            order = np.argsort(-deficit)
            allocated = np.zeros(k, dtype=int)
            rem = remaining
            for idx in order:
                take = min(int(deficit[idx]), rem)
                allocated[idx] = take
                rem -= take
                if rem <= 0:
                    break
            deficit = allocated

        if deficit.sum() == 0:
            # No new samples requested by OCBA rule; force at least
            # one sample to the system with smallest current count.
            least = int(np.argmin(current))
            deficit[least] = 1

        # Take the samples (each using a fresh CRN draw)
        for i in range(k):
            for _ in range(int(deficit[i])):
                inputs = generate_common_inputs(base_seed, global_rep)
                samples[i].append(sim_fn_Y(i, inputs))
                global_rep += 1
                used += 1
                if used >= budget:
                    break
            if used >= budget:
                break
        iteration += 1

        if verbose and iteration % 5 == 0:
            print(f"    iter {iteration}: used {used}/{budget}")

    # Final selection
    final_means = np.array([np.mean(samples[i]) for i in range(k)])
    selected = int(np.argmin(final_means))
    sample_counts = np.array([len(samples[i]) for i in range(k)])
    return {
        "selected": selected,
        "final_means": final_means,
        "sample_counts": sample_counts,
        "budget_used": used,
        "n0": n0,
    }


def run_comparison(n_macro_reps: int = 5,
                   known_best_label_prefix: str = "P04"):
    """
    Head-to-head comparison of KN (with CRN) vs OCBA (classical) across
    n_macro_reps independent base seeds. Reports empirical P(CS) and the
    per-system sample distribution of OCBA averaged across macro reps.
    """
    print("=" * 70)
    print("OCBA vs KN COMPARISON")
    print("=" * 70)
    print(f"  Macro reps:         {n_macro_reps}")
    print(f"  k:                  {N_PORTFOLIOS}")
    print(f"  n0 per system:      {KN_N0}")

    known_best_idx = next(
        p.pid for p in PORTFOLIOS if p.label.startswith(known_best_label_prefix)
    )
    print(f"  Ground-truth best:  {PORTFOLIOS[known_best_idx].label} "
          f"(from baseline KN with base_seed=42)")
    print()

    kn_calls_log = []
    kn_correct = 0
    ocba_correct = 0
    ocba_alloc_matrix = np.zeros((n_macro_reps, N_PORTFOLIOS), dtype=int)
    kn_alloc_matrix = np.zeros((n_macro_reps, N_PORTFOLIOS), dtype=int)

    t_start = time.time()
    for m in range(n_macro_reps):
        seed = 100 + m
        # --- KN run ---
        t1 = time.time()
        kn_res = kn_procedure(sim_fn_Y, generate_common_inputs, base_seed=seed)
        kn_time = time.time() - t1
        kn_calls = sum(len(v) for v in kn_res["all_data"].values())
        kn_calls_log.append(kn_calls)
        if kn_res["selected"] == known_best_idx:
            kn_correct += 1
        for i in range(N_PORTFOLIOS):
            kn_alloc_matrix[m, i] = len(kn_res["all_data"][i])

        # --- OCBA run to the same budget ---
        t2 = time.time()
        ocba_res = run_ocba(budget=kn_calls, base_seed=seed)
        ocba_time = time.time() - t2
        if ocba_res["selected"] == known_best_idx:
            ocba_correct += 1
        ocba_alloc_matrix[m, :] = ocba_res["sample_counts"]

        print(f"  [macro {m+1}/{n_macro_reps}] seed={seed}  "
              f"KN used {kn_calls} calls ({kn_time:.0f}s) -> "
              f"{PORTFOLIOS[kn_res['selected']].label}; "
              f"OCBA used {ocba_res['budget_used']} calls ({ocba_time:.0f}s) "
              f"-> {PORTFOLIOS[ocba_res['selected']].label}")

    total_time = time.time() - t_start
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  KN   empirical P(CS) = {kn_correct}/{n_macro_reps} "
          f"= {kn_correct/n_macro_reps*100:.0f}%")
    print(f"  OCBA empirical P(CS) = {ocba_correct}/{n_macro_reps} "
          f"= {ocba_correct/n_macro_reps*100:.0f}%")
    print(f"  KN mean calls per run:   {np.mean(kn_calls_log):.0f}")
    print(f"  Total wall clock: {total_time:.0f}s")

    # Per-system average allocation
    avg_kn = kn_alloc_matrix.mean(axis=0)
    avg_ocba = ocba_alloc_matrix.mean(axis=0)
    print()
    print(f"  Average samples per portfolio (ordered by KN baseline rank):")
    print(f"  {'Portfolio':<22} {'KN avg':>10} {'OCBA avg':>10} "
          f"{'OCBA frac':>12}")
    print(f"  {'-'*58}")
    total_ocba = avg_ocba.sum()
    # Order portfolios by mean Y in the baseline KN run for readability
    base_kn = kn_procedure(sim_fn_Y, generate_common_inputs, base_seed=42)
    baseline_means = {
        i: np.mean(base_kn["all_data"][i]) for i in range(N_PORTFOLIOS)
    }
    order = sorted(range(N_PORTFOLIOS), key=lambda i: baseline_means[i])
    for i in order:
        print(f"  {PORTFOLIOS[i].label:<22} {avg_kn[i]:>10.1f} "
              f"{avg_ocba[i]:>10.1f} {avg_ocba[i]/total_ocba*100:>11.1f}%")

    return {
        "kn_correct": kn_correct,
        "ocba_correct": ocba_correct,
        "n_macro_reps": n_macro_reps,
        "kn_calls_log": kn_calls_log,
        "avg_kn_alloc": avg_kn,
        "avg_ocba_alloc": avg_ocba,
        "known_best": known_best_idx,
    }


if __name__ == "__main__":
    run_comparison(n_macro_reps=5)
