#!/usr/bin/env python3
"""
Generate Figures for Rebuttal Paper
======================================
Creates publication-quality figures from experiment results:
1. Cross-domain recall ceiling bar chart (8 datasets)
2. Catalog size / density vs recall scatter plots
3. Density subsampling curves
4. Prompt quality × recall interaction
5. Reranker comparison across datasets

Usage:
    python scripts/generate_rebuttal_figures.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

fig_dir = project_root / 'submissions' / 'rebuttal' / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)
logs = project_root / 'experiments' / 'logs'


def load_json(name):
    with open(logs / name) as f:
        return json.load(f)


# ============================================================================
# Figure 1: Cross-Domain Recall Ceiling (8 datasets)
# ============================================================================
def fig_cross_domain_recall():
    unified = load_json('unified_experiment_results.json')

    datasets = list(unified.keys())
    recalls = [unified[d]['best_recall_100']['recall'] * 100 for d in datasets]
    zero_pcts = [unified[d]['raep']['zero_recall_pct'] for d in datasets]
    domains = [unified[d]['stats']['domain'] for d in datasets]

    colors = {'product': '#4C72B0', 'news': '#DD8452', 'movie': '#55A868'}
    bar_colors = [colors[d] for d in domains]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Recall@100
    bars = ax1.bar(range(len(datasets)), recalls, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.set_ylabel('Recall@100 (%)')
    ax1.set_title('(a) Best Retrieval Recall@100')
    ax1.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Viability threshold (15%)')
    ax1.legend(loc='upper left', framealpha=0.8)

    # Zero-recall percentage
    ax2.bar(range(len(datasets)), zero_pcts, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.set_ylabel('Zero-Recall Users (%)')
    ax2.set_title('(b) Users with No Relevant Candidates')

    # Legend for domains
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=d.capitalize()) for d, c in colors.items()]
    ax2.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig_cross_domain_recall.pdf')
    plt.savefig(fig_dir / 'fig_cross_domain_recall.png')
    plt.close()
    print("  fig_cross_domain_recall.pdf")


# ============================================================================
# Figure 2: Density vs Recall Scatter
# ============================================================================
def fig_density_scatter():
    catalog = load_json('catalog_size_analysis.json')
    ds_data = catalog['datasets']

    names = list(ds_data.keys())
    densities = [ds_data[n]['density'] * 100 for n in names]
    catalogs = [ds_data[n]['catalog_size'] for n in names]
    recall_100 = [ds_data[n]['recall_at_K']['K=100'] * 100 for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Density vs Recall
    ax1.scatter(densities, recall_100, s=80, c='#4C72B0', edgecolors='white', linewidth=0.5, zorder=3)
    for i, n in enumerate(names):
        ax1.annotate(n, (densities[i], recall_100[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Fit line
    log_d = np.log10(np.array(densities))
    r = np.array(recall_100)
    z = np.polyfit(log_d, r, 1)
    x_fit = np.linspace(min(log_d), max(log_d), 100)
    ax1.plot(10**x_fit, np.polyval(z, x_fit), 'r--', alpha=0.5)

    corr = catalog['analysis']['density_vs_recall']['pearson_r']
    ax1.set_xlabel('Interaction Density (%)')
    ax1.set_xscale('log')
    ax1.set_ylabel('Recall@100 (%)')
    ax1.set_title(f'(a) Density vs Recall (r={corr:.2f})')
    ax1.grid(True, alpha=0.3)

    # Catalog size vs Recall
    ax2.scatter(catalogs, recall_100, s=80, c='#DD8452', edgecolors='white', linewidth=0.5, zorder=3)
    for i, n in enumerate(names):
        ax2.annotate(n, (catalogs[i], recall_100[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    corr_c = catalog['analysis']['catalog_vs_recall']['pearson_r']
    ax2.set_xlabel('Catalog Size')
    ax2.set_xscale('log')
    ax2.set_ylabel('Recall@100 (%)')
    ax2.set_title(f'(b) Catalog Size vs Recall (r={corr_c:.2f})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig_density_catalog_scatter.pdf')
    plt.savefig(fig_dir / 'fig_density_catalog_scatter.png')
    plt.close()
    print("  fig_density_catalog_scatter.pdf")


# ============================================================================
# Figure 3: Density Subsampling Curves
# ============================================================================
def fig_density_curves():
    try:
        density = load_json('density_recall_analysis.json')
    except FileNotFoundError:
        print("  SKIP: density_recall_analysis.json not found")
        return

    ds_data = density['datasets']

    fig, ax = plt.subplots(figsize=(6, 4))

    colors_list = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
    for i, (key, res) in enumerate(ds_data.items()):
        points = [(r['density'] * 100, r['recall'] * 100)
                  for r in res['results'] if r['K'] == 100]
        if len(points) < 2:
            continue
        points.sort(key=lambda x: x[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.plot(x, y, 'o-', color=colors_list[i % len(colors_list)],
                label=key, markersize=6, linewidth=1.5)

    ax.set_xlabel('Interaction Density (%)')
    ax.set_xscale('log')
    ax.set_ylabel('Recall@100 (%)')
    ax.set_title('Recall vs Density (Subsampling Analysis)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig_density_subsampling.pdf')
    plt.savefig(fig_dir / 'fig_density_subsampling.png')
    plt.close()
    print("  fig_density_subsampling.pdf")


# ============================================================================
# Figure 4: Prompt Quality × Recall
# ============================================================================
def fig_prompt_recall():
    try:
        prompt = load_json('prompt_complexity_analysis.json')
    except FileNotFoundError:
        print("  SKIP: prompt_complexity_analysis.json not found")
        return

    ds_data = prompt['datasets']

    # Pick one representative dataset
    key = 'beauty' if 'beauty' in ds_data else list(ds_data.keys())[0]
    levels = ds_data[key]['recall_levels']

    recalls = [l['actual_recall'] * 100 for l in levels]
    oracle = [l['strategies']['oracle']['ndcg_mean'] for l in levels]
    cf = [l['strategies']['cf_order']['ndcg_mean'] for l in levels]
    random_ = [l['strategies']['random']['ndcg_mean'] for l in levels]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(recalls, oracle, 's-', color='#55A868', label='Oracle (perfect prompt)', linewidth=1.5)
    ax.plot(recalls, cf, 'o-', color='#4C72B0', label='CF-order (basic prompt)', linewidth=1.5)
    ax.plot(recalls, random_, '^-', color='#C44E52', label='Random (bad prompt)', linewidth=1.5)

    # Shade realistic region
    ax.axvspan(0, 15, alpha=0.1, color='red', label='Realistic recall region')

    ax.set_xlabel('Recall@100 (%)')
    ax.set_ylabel('NDCG@10')
    ax.set_title(f'Prompt Quality Effect vs Recall Level ({key})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig_prompt_vs_recall.pdf')
    plt.savefig(fig_dir / 'fig_prompt_vs_recall.png')
    plt.close()
    print("  fig_prompt_vs_recall.pdf")


# ============================================================================
# Figure 5: Recall@K across all datasets (K sensitivity)
# ============================================================================
def fig_recall_at_k():
    catalog = load_json('catalog_size_analysis.json')
    ds_data = catalog['datasets']

    fig, ax = plt.subplots(figsize=(7, 4.5))

    K_values = [20, 50, 100, 200, 500]
    colors_map = {
        'beauty': '#4C72B0', 'movies': '#DD8452', 'electronics': '#55A868',
        'toys': '#C44E52', 'sports': '#8172B2', 'office': '#CCB974',
        'mind_news': '#64B5CD', 'movielens': '#000000'
    }
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    for i, (name, res) in enumerate(ds_data.items()):
        recalls = [res['recall_at_K'].get(f'K={k}', 0) * 100 for k in K_values]
        ax.plot(K_values, recalls,
                marker=markers[i % len(markers)],
                color=colors_map.get(name, f'C{i}'),
                label=f"{name} ({res['catalog_size']:,})",
                linewidth=1.5, markersize=5)

    ax.set_xlabel('Candidate Set Size (K)')
    ax.set_ylabel('Recall@K (%)')
    ax.set_title('Recall@K Across 8 Datasets')
    ax.set_xscale('log')
    ax.set_xticks(K_values)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig_recall_at_k.pdf')
    plt.savefig(fig_dir / 'fig_recall_at_k.png')
    plt.close()
    print("  fig_recall_at_k.pdf")


# ============================================================================
# Main
# ============================================================================
def main():
    print("Generating rebuttal figures...")
    print(f"Output: {fig_dir}\n")

    fig_cross_domain_recall()
    fig_density_scatter()
    fig_density_curves()
    fig_prompt_recall()
    fig_recall_at_k()

    print(f"\nAll figures saved to {fig_dir}")


if __name__ == '__main__':
    main()
