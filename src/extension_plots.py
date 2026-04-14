"""
extension_plots.py -- Figures for the four methodological extensions.

Generates:
  - fig_elcc_empirical.png       (empirical vs static ELCC bar chart)
  - fig_ocba_allocation.png      (KN vs OCBA sample allocation profile)
  - fig_capital_sensitivity.png  (capital cost sensitivity tornado)

Run: python src/extension_plots.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 11

from config import PORTFOLIOS, CAPITAL_COST
from capital_sensitivity import (
    run_capital_sensitivity, _collect_baseline_from_pipeline,
)
from empirical_elcc import run_empirical_elcc
from ocba import run_comparison as run_ocba_comparison

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")


def plot_empirical_elcc(elcc_result, out_path):
    """Bar chart comparing empirical vs static ELCC."""
    techs = ["Wind", "Solar", "Storage (4h)", "Realistic firm"]
    keys = ["wind", "solar", "storage", "realistic_firm"]
    emp = [elcc_result["empirical_elcc"][k] for k in keys]
    static = [elcc_result["static_elcc"][k] for k in keys]

    x = np.arange(len(techs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    b1 = ax.bar(x - w/2, static, w, label="Static (planning)",
                color="#6a8caf", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, emp, w, label="Empirical (simulated)",
                color="#d46a5f", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("ELCC (fraction of installed MW)")
    ax.set_xticks(x)
    ax.set_xticklabels(techs)
    ax.set_ylim(0, 1.05)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    # Annotate bars with values
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Empirical vs. static ELCC (1,000 MW dose on stressed baseline)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_ocba_allocation(ocba_result, portfolios, out_path):
    """Bar chart of average samples per portfolio under KN vs OCBA."""
    k = len(portfolios)
    # Order portfolios by KN baseline mean Y (best first) for readability
    from kn import kn_procedure
    from crn import generate_common_inputs
    from ocba import sim_fn_Y
    base = kn_procedure(sim_fn_Y, generate_common_inputs, base_seed=42)
    means = {i: np.mean(base["all_data"][i]) for i in range(k)}
    order = sorted(range(k), key=lambda i: means[i])
    labels = [portfolios[i].label.split(" ")[0] for i in order]  # Just "P04"

    kn_alloc = ocba_result["avg_kn_alloc"][order]
    ocba_alloc = ocba_result["avg_ocba_alloc"][order]

    x = np.arange(k)
    w = 0.40
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(x - w/2, kn_alloc, w, label="KN (with CRN)",
           color="#6a8caf", edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, ocba_alloc, w, label="OCBA (independent)",
           color="#d46a5f", edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Avg. samples (5 macro-reps)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    # Highlight the ground-truth best
    best_idx_in_order = 0  # Already first because we sorted
    ax.axvspan(best_idx_in_order - 0.5, best_idx_in_order + 0.5,
               color="#ffd966", alpha=0.25, zorder=0)
    ax.text(best_idx_in_order, max(max(kn_alloc), max(ocba_alloc)) * 0.95,
            "ground truth", ha="center", va="top", fontsize=9,
            color="#8b6914", style="italic")

    ax.set_title("R&S sample allocation: KN elimination vs. OCBA concentration")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_capital_sensitivity(cap_result, mean_Y, top_k, out_path):
    """Tornado plot of the P04-to-P10 gap under +/-30% cost perturbations."""
    # For each cost parameter, compute the gap (P10 - P04) at -30%, 0, +30%
    # and draw a horizontal bar showing the range.
    # Baseline gap
    p04_idx = top_k[0]
    p10_idx = top_k[1]
    base_gap = mean_Y[p10_idx] - mean_Y[p04_idx]

    params = list(cap_result.keys())
    # For each param, find min and max P10-P04 gap across its 5 perturbations
    ranges = []
    for param in params:
        gaps = []
        for cfg in cap_result[param]["sweep"]:
            means = cfg["means"]
            gap = means[p10_idx] - means[p04_idx]
            gaps.append((cfg["pct"], gap))
        gaps.sort()
        gmin = min(g[1] for g in gaps)
        gmax = max(g[1] for g in gaps)
        ranges.append((param, gmin, gmax))

    # Sort by sensitivity magnitude (widest range first)
    ranges.sort(key=lambda r: r[2] - r[1], reverse=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    names = [r[0] for r in ranges]
    lows = np.array([r[1]/1e6 for r in ranges])
    highs = np.array([r[2]/1e6 for r in ranges])
    y = np.arange(len(names))
    ax.barh(y, highs - lows, left=lows, height=0.55,
            color="#6a8caf", edgecolor="black", linewidth=0.5,
            label="P10 $-$ P04 gap under $\\pm 30\\%$")
    # Baseline line
    ax.axvline(base_gap / 1e6, color="#d46a5f", linestyle="--",
               linewidth=1.2, label=f"Baseline gap = \\${base_gap/1e6:.0f}M")
    # Indifference zone
    ax.axvline(50, color="grey", linestyle=":", linewidth=1.0,
               label=r"Indifference zone $\delta = \$50$M")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Highest sensitivity on top
    ax.set_xlabel(r"P10 $-$ P04 Mean $Y$ gap (\$M)")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_title(r"Capital cost sensitivity: P04 remains best across all $\pm 30\%$ sweeps")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    print("[1/3] Running empirical ELCC (100 reps)...")
    elcc_result = run_empirical_elcc(baseline_firm_mw=5000.0, dose_mw=1000.0, n_reps=100)
    plot_empirical_elcc(elcc_result, os.path.join(FIG_DIR, "fig_elcc_empirical.png"))

    print("\n[2/3] Running capital cost sensitivity...")
    top_k, mean_Y, mean_F = _collect_baseline_from_pipeline(n_reps=200)
    cap_result = run_capital_sensitivity(top_k, mean_Y, mean_F, verbose=False)
    plot_capital_sensitivity(cap_result, mean_Y, top_k,
                              os.path.join(FIG_DIR, "fig_capital_sensitivity.png"))

    print("\n[3/3] Running OCBA vs KN comparison (5 macro reps)...")
    ocba_result = run_ocba_comparison(n_macro_reps=5)
    plot_ocba_allocation(ocba_result, PORTFOLIOS,
                          os.path.join(FIG_DIR, "fig_ocba_allocation.png"))

    print("\nDone. Figures written to", FIG_DIR)
