"""
structural_plots.py -- Figures for the boundary and capacity-sweep probes.

Generates two figures for the new Structural Sensitivity section:
  fig_boundary.png     -- Mean Y vs firm share across boundary portfolios
  fig_capacity_sweep.png -- Mean Y of top-4 vs total capacity (12/15.8/18 GW)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 11

from config import PORTFOLIOS, Portfolio, TOTAL_CAPACITY_MW
from crn import generate_common_inputs
from dispatch import simulate_replication

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")

N_REPS = 100


def _evaluate(p, n_reps=N_REPS):
    Y = []
    for rep in range(n_reps):
        inp = generate_common_inputs(base_seed=42, replication=rep)
        _, _, _, Y_ij = simulate_replication(p, inp)
        Y.append(Y_ij)
    return float(np.mean(Y)), float(np.std(Y, ddof=1) / np.sqrt(n_reps))


def _make(pid, firm, renew, storage, total_mw=TOTAL_CAPACITY_MW):
    return Portfolio(
        pid=pid, firm_frac=firm, renew_frac=renew,
        storage_frac=storage, total_capacity_mw=total_mw,
    )


def plot_boundary(out_path):
    """Bar chart of Mean Y at 12 GW across portfolios that span the
    70% firm upper bound into unexplored territory."""
    print("[1/2] Boundary probe figure...")
    candidates = [
        ("P03\n(60/30/10)", _make(9001, 0.60, 0.30, 0.10)),
        ("P04\n(70/20/10)", _make(9002, 0.70, 0.20, 0.10)),
        ("P$_{b1}$\n(75/15/10)", _make(9003, 0.75, 0.15, 0.10)),
        ("P$_{b4}$\n(80/15/05)", _make(9004, 0.80, 0.15, 0.05)),
        ("P$_{b2}$\n(80/10/10)", _make(9005, 0.80, 0.10, 0.10)),
        ("P$_{b3}$\n(85/10/05)", _make(9006, 0.85, 0.10, 0.05)),
    ]
    labels, means, sems = [], [], []
    for lbl, p in candidates:
        my, sem = _evaluate(p)
        labels.append(lbl)
        means.append(my / 1e6)
        sems.append(sem / 1e6)

    x = np.arange(len(labels))
    colors = ["#cccccc", "#cccccc", "#ffd966", "#ffd966", "#ffd966", "#d46a5f"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bars = ax.bar(x, means, yerr=sems, capsize=3,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean $Y$ (\\$M)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    # Annotate with values
    for rect, val in zip(bars, means):
        ax.text(rect.get_x() + rect.get_width() / 2,
                val + max(means) * 0.015,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9)
    # Shade the "original feasible region" bar group
    ax.axvspan(-0.5, 1.5, color="#e8e8e8", alpha=0.3, zorder=0)
    ax.text(0.5, max(means) * 0.97, "original simplex bounds",
            ha="center", va="top", fontsize=8, style="italic", color="#555555")
    ax.set_title("Boundary probe at 12 GW: firm share beyond the 70\\% upper bound")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_capacity_sweep(out_path):
    """Grouped bar chart of Mean Y for the top-4 portfolios at three
    total-capacity levels: 12 GW, 15.8 GW, 18 GW."""
    print("[2/2] Capacity-sweep figure...")
    top4 = [
        ("P04\n(70/20/10)", 0.70, 0.20, 0.10, "#d46a5f"),
        ("P10\n(70/10/20)", 0.70, 0.10, 0.20, "#e4947a"),
        ("P03\n(60/30/10)", 0.60, 0.30, 0.10, "#6a8caf"),
        ("P09\n(60/20/20)", 0.60, 0.20, 0.20, "#9bb4cb"),
    ]
    capacities = [(12_000, "12 GW\n(inadequate)"),
                  (15_800, "15.8 GW\n(P04 adequate)"),
                  (18_000, "18 GW\n(all adequate)")]
    results = np.zeros((len(top4), len(capacities)))
    for j, (C, _) in enumerate(capacities):
        for i, (_, f, r, s, _) in enumerate(top4):
            p = _make(9100 + i * 10 + j, f, r, s, total_mw=C)
            my, _ = _evaluate(p)
            results[i, j] = my / 1e6

    x = np.arange(len(capacities))
    w = 0.20
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    for i, (lbl, _, _, _, color) in enumerate(top4):
        offset = (i - (len(top4) - 1) / 2) * w
        bars = ax.bar(x + offset, results[i], w, label=lbl,
                      color=color, edgecolor="black", linewidth=0.4)
        # Annotate winner with a star
        for j in range(len(capacities)):
            if results[i, j] == results[:, j].min():
                ax.text(x[j] + offset, results[i, j] + 200,
                        "$\\star$", ha="center", va="bottom",
                        fontsize=14, color="#1a6a1a")
    ax.set_ylabel("Mean $Y$ (\\$M)")
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in capacities], fontsize=9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, ncol=2, fontsize=9)
    ax.set_title("Top-4 portfolios across three total-capacity levels ($\\star$ = winner)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_boundary(os.path.join(FIG_DIR, "fig_boundary.png"))
    plot_capacity_sweep(os.path.join(FIG_DIR, "fig_capacity_sweep.png"))
    print("Done.")
