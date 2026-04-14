"""
plots.py — Generate all figures for the report.

Figures:
1. Bar chart of mean Y with 95% CIs (top-k)
2. KN elimination timeline
3. Box plots of Y distributions (top-k)
4. Tornado diagram of factor importance
5. Cumulative mean convergence plot
6. DBG comparison bar chart
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Consistent style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def generate_all_plots(results, portfolios):
    """Generate all report figures from experiment results."""
    fig1_mean_y_bar(results, portfolios)
    fig2_kn_elimination(results, portfolios)
    fig3_y_distributions(results, portfolios)
    fig4_tornado(results, portfolios)
    fig5_convergence(results, portfolios)
    fig6_dbg_comparison(results, portfolios)
    print(f"  Generated 6 figures in {FIGURE_DIR}/")


def fig1_mean_y_bar(results, portfolios):
    """Bar chart of mean Y with 95% CIs for top-k portfolios."""
    ci = results['ci']['individual_cis']
    top_k = results['top_k']

    labels = [portfolios[i].label for i in top_k]
    means = [ci[i]['mean'] / 1e6 for i in top_k]
    errors = [ci[i]['half_width'] / 1e6 for i in top_k]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(top_k)), means, yerr=errors,
                  capsize=5, color=COLORS[:len(top_k)], edgecolor='black',
                  linewidth=0.5, alpha=0.85)

    # Annotate bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errors)*0.5,
                f'${mean:,.0f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(top_k)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Mean Annual Cost ($M)')
    ax.set_title('Portfolio Performance: Mean $Y_i$ with 95% Confidence Intervals')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}M'))
    ax.set_ylim(0, max(means) * 1.15)

    # Mark the KN-selected best
    best_idx_in_topk = top_k.index(results['kn']['selected'])
    bars[best_idx_in_topk].set_edgecolor('red')
    bars[best_idx_in_topk].set_linewidth(2.5)

    fig.savefig(os.path.join(FIGURE_DIR, 'fig1_mean_y_bar.png'))
    plt.close(fig)


def fig2_kn_elimination(results, portfolios):
    """KN elimination timeline showing when each portfolio was dropped."""
    log = results['kn']['elimination_log']
    if not log:
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    # Collect all annotation positions for overlap avoidance
    annot_positions = []
    for i, (stage, pid, mean, _, _) in enumerate(log):
        annot_positions.append((stage, mean / 1e6, portfolios[pid].label))

    # Sort by y-value within the same stage to stagger labels
    from collections import defaultdict
    stage_groups = defaultdict(list)
    for stage, y, label in annot_positions:
        stage_groups[stage].append((y, label))

    for i, (stage, pid, mean, _, _) in enumerate(log):
        ax.scatter(stage, mean / 1e6, s=80, zorder=5, color=COLORS[i % len(COLORS)])

    # Stagger labels per stage to avoid overlap
    for stage, items in stage_groups.items():
        items.sort(key=lambda t: t[0], reverse=True)
        for rank, (y, label) in enumerate(items):
            x_off = 10 + (rank % 2) * 30
            y_off = 5 - rank * 2
            ax.annotate(label, (stage, y),
                        textcoords="offset points", xytext=(x_off, y_off),
                        fontsize=8, ha='left',
                        arrowprops=dict(arrowstyle='-', color='gray',
                                        lw=0.5, shrinkA=0, shrinkB=3))

    # Mark the selected best
    best = results['kn']['selected']
    final_mean = results['kn']['final_means'][best] / 1e6
    ax.scatter(results['kn']['total_reps'], final_mean,
               s=150, marker='*', color='red', zorder=10, label='Selected')
    ax.annotate(f'{portfolios[best].label}\n(Selected)',
                (results['kn']['total_reps'], final_mean),
                textcoords="offset points", xytext=(10, -10),
                fontsize=10, fontweight='bold', color='red')

    ax.set_xlabel('Replication Stage', fontsize=12)
    ax.set_ylabel('Sample Mean ($M)', fontsize=12)
    ax.set_title('Kim-Nelson Elimination Timeline', fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}M'))

    fig.savefig(os.path.join(FIGURE_DIR, 'fig2_kn_elimination.png'))
    plt.close(fig)


def fig3_y_distributions(results, portfolios):
    """Box plots of Y distributions for top-k portfolios."""
    top_k = results['top_k']
    Y = results['extended_eval']['Y']

    fig, ax = plt.subplots(figsize=(8, 5))

    data = [np.array(Y[i]) / 1e6 for i in top_k]
    labels = [portfolios[i].label for i in top_k]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    for patch, color in zip(bp['boxes'], COLORS[:len(top_k)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Annual Cost ($M)')
    ax.set_title('Distribution of $Y_{ij}$ Across Replications (n=200)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}M'))

    fig.savefig(os.path.join(FIGURE_DIR, 'fig3_y_distributions.png'))
    plt.close(fig)


def fig4_tornado(results, portfolios):
    """Tornado diagram of factor importance."""
    main_effects = results['factorial']['main_effects']

    factors = sorted(main_effects.keys(), key=lambda k: abs(main_effects[k]))
    values = [main_effects[f] / 1e6 for f in factors]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in values]
    ax.barh(range(len(factors)), values, color=colors, edgecolor='black',
            linewidth=0.5, alpha=0.85)

    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(factors, fontsize=10)
    ax.set_xlabel('Effect on Best-Worst Performance Gap ($M)')
    ax.set_title('Factor Importance: Main Effects (2³ Factorial)')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}M'))

    fig.savefig(os.path.join(FIGURE_DIR, 'fig4_tornado.png'))
    plt.close(fig)


def fig5_convergence(results, portfolios):
    """Cumulative mean convergence plot for top-k portfolios."""
    top_k = results['top_k']
    Y = results['extended_eval']['Y']
    n = results['extended_eval']['n_reps']

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, color in zip(top_k, COLORS):
        y = np.array(Y[idx])
        cum_mean = np.cumsum(y) / np.arange(1, len(y) + 1)
        ax.plot(range(1, len(y) + 1), cum_mean / 1e6,
                label=portfolios[idx].label, color=color, linewidth=1.5)

    ax.set_xlabel('Number of Replications')
    ax.set_ylabel('Cumulative Mean ($M)')
    ax.set_title('Convergence of Sample Mean $\\bar{Y}_i(r)$')
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}M'))

    fig.savefig(os.path.join(FIGURE_DIR, 'fig5_convergence.png'))
    plt.close(fig)


def fig6_dbg_comparison(results, portfolios):
    """DBG comparison: deterministic vs stochastic recommendation."""
    dbg = results['dbg']

    if dbg['same_portfolio']:
        # Still generate a simple bar chart of deterministic evaluations
        fig, ax = plt.subplots(figsize=(8, 5))
        det_Y = dbg['det_Y']
        sorted_items = sorted(det_Y.items(), key=lambda x: x[1])[:6]
        labels = [portfolios[i].label for i, _ in sorted_items]
        vals = [y / 1e6 for _, y in sorted_items]
        ax.bar(range(len(labels)), vals, color=COLORS[:len(labels)],
               edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.set_ylabel('Deterministic Cost ($M)')
        ax.set_title('Deterministic Evaluation (Same Best as Stochastic)')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}M'))
        fig.savefig(os.path.join(FIGURE_DIR, 'fig6_dbg.png'))
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    labels = [
        f"Det: {portfolios[dbg['det_best']].label}",
        f"KN: {portfolios[dbg['kn_selected']].label}",
    ]
    means = [dbg['Y_det_mean'] / 1e6, dbg['Y_kn_mean'] / 1e6]
    colors_bar = ['#d62728', '#2ca02c']

    bars = ax.bar(range(2), means, color=colors_bar, edgecolor='black',
                  linewidth=0.5, alpha=0.85, width=0.5)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'${mean:,.0f}M', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Mean Annual Cost Under Uncertainty ($M)')
    ax.set_title(f'Deterministic Benchmark Gap: ${dbg["dbg"]/1e6:,.0f}M')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}M'))

    fig.savefig(os.path.join(FIGURE_DIR, 'fig6_dbg.png'))
    plt.close(fig)
