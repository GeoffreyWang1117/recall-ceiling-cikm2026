#!/usr/bin/env python3
"""
Generate publication-quality figures for KDD 2026 paper.
All figures use consistent styling suitable for ACM proceedings.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ============================================================================
# Style Configuration (ACM/KDD compatible)
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

PROJECT_ROOT = Path(__file__).parent.parent
LOGS = PROJECT_ROOT / 'experiments' / 'logs'
OUTPUT = PROJECT_ROOT / 'submissions' / 'kdd2026' / 'figures'
OUTPUT.mkdir(exist_ok=True)

COLORS = {
    'primary': '#2171B5',     # blue
    'secondary': '#CB181D',   # red
    'accent1': '#238B45',     # green
    'accent2': '#6A3D9A',     # purple
    'accent3': '#FF7F00',     # orange
    'light': '#9ECAE1',       # light blue
    'gray': '#636363',
    'cf_baseline': '#333333',
}


def load_json(name):
    with open(LOGS / name) as f:
        return json.load(f)


# ============================================================================
# Figure 1: Catalog Size vs Best Recall@100
# ============================================================================
def figure1_catalog_vs_recall():
    """Scatter plot: catalog size determines retrieval feasibility."""
    rb = load_json('kdd_retrieval_baselines.json')

    # Core datasets: best recall per dataset
    # Catalog sizes from oracle data
    orc = load_json('kdd_oracle_realistic_n500.json')
    catalog_sizes = {
        'beauty': orc['datasets']['amazon_beauty']['n_items'],
        'movies': orc['datasets']['amazon_movies']['n_items'],
        'electronics': orc['datasets']['amazon_electronics']['n_items'],
    }
    datasets = []
    for ds_name in ['beauty', 'movies', 'electronics']:
        ds = rb[ds_name]
        best_recall = 0
        best_method = ''
        best_ci_lo, best_ci_hi = 0, 0
        for method, data in ds['results'].items():
            recall_mean = data['mean']
            if recall_mean > best_recall:
                best_recall = recall_mean
                best_method = method
                best_ci_lo = data['ci_low']
                best_ci_hi = data['ci_high']
        datasets.append({
            'name': ds_name.capitalize(),
            'catalog': catalog_sizes[ds_name],
            'recall': best_recall * 100,
            'ci_lo': best_ci_lo * 100,
            'ci_hi': best_ci_hi * 100,
        })

    # Large-scale datasets
    for ds_file, ds_label in [('kdd_large_scale_sports.json', 'Sports'),
                               ('kdd_large_scale_toys.json', 'Toys'),
                               ('kdd_large_scale_home_kitchen.json', 'Home')]:
        ls = load_json(ds_file)
        best_recall = 0
        best_ci_lo, best_ci_hi = 0, 0
        for method, data in ls['retrieval_results'].items():
            r = data['recall@100']['mean'] * 100
            if r > best_recall:
                best_recall = r
                best_ci_lo = data['recall@100']['ci_low'] * 100
                best_ci_hi = data['recall@100']['ci_high'] * 100
        datasets.append({
            'name': ds_label,
            'catalog': ls['config']['n_items'],
            'recall': best_recall,
            'ci_lo': best_ci_lo,
            'ci_hi': best_ci_hi,
        })

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    catalogs = [d['catalog'] for d in datasets]
    recalls = [d['recall'] for d in datasets]
    err_lo = [d['recall'] - d['ci_lo'] for d in datasets]
    err_hi = [d['ci_hi'] - d['recall'] for d in datasets]
    names = [d['name'] for d in datasets]

    ax.errorbar(catalogs, recalls, yerr=[err_lo, err_hi],
                fmt='o', markersize=8, color=COLORS['primary'],
                ecolor=COLORS['gray'], elinewidth=1.2, capsize=4, capthick=1.2,
                markeredgecolor='white', markeredgewidth=0.8, zorder=5)

    for i, name in enumerate(names):
        offset_x = 1.15 if name != 'Beauty' else 0.85
        offset_y = 0.3 if name not in ['Toys', 'Home'] else -0.5
        if name == 'Toys':
            offset_y = -0.6
        ax.annotate(name, (catalogs[i], recalls[i]),
                    xytext=(catalogs[i] * offset_x, recalls[i] + offset_y),
                    fontsize=8, ha='left' if offset_x > 1 else 'right')

    # Add shaded region for "reranking viable" zone
    ax.axhspan(5, 12, alpha=0.08, color=COLORS['accent1'], zorder=0)
    ax.text(1200, 10.5, 'Reranking\nviable zone', fontsize=7,
            color=COLORS['accent1'], alpha=0.7, style='italic')
    ax.axhline(y=5, color=COLORS['accent1'], linestyle=':', alpha=0.4)

    ax.set_xscale('log')
    ax.set_xlabel('Catalog Size (number of items)')
    ax.set_ylabel('Best Recall@100 (%)')
    ax.set_xlim(800, 80000)
    ax.set_ylim(0, 12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{int(x):,}' if x >= 1000 else str(int(x))))

    fig.savefig(OUTPUT / 'fig_catalog_recall.pdf')
    fig.savefig(OUTPUT / 'fig_catalog_recall.png')
    print(f"Figure 1 saved: {OUTPUT / 'fig_catalog_recall.pdf'}")
    plt.close(fig)


# ============================================================================
# Figure 2: Oracle vs Realistic Gap
# ============================================================================
def figure2_oracle_realistic_gap():
    """Grouped bar chart showing the massive oracle-realistic gap."""
    orc = load_json('kdd_oracle_realistic_n500.json')

    ds_names = ['Beauty', 'Movies', 'Electronics']
    ds_keys = ['amazon_beauty', 'amazon_movies', 'amazon_electronics']

    oracle_means = []
    oracle_cis = []
    realistic_means = []
    realistic_cis = []
    gaps = []

    for key in ds_keys:
        d = orc['datasets'][key]
        o_mean = d['oracle']['mean']
        o_ci_lo = d['oracle']['ci_low']
        o_ci_hi = d['oracle']['ci_high']
        r_mean = d['realistic']['mean']
        r_ci_lo = d['realistic']['ci_low']
        r_ci_hi = d['realistic']['ci_high']

        oracle_means.append(o_mean)
        oracle_cis.append([o_mean - o_ci_lo, o_ci_hi - o_mean])
        realistic_means.append(r_mean)
        realistic_cis.append([r_mean - r_ci_lo, r_ci_hi - r_mean])
        gaps.append(d['gap_percent'])

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    x = np.arange(len(ds_names))
    width = 0.32

    bars1 = ax.bar(x - width/2, oracle_means, width, label='Oracle',
                   color=COLORS['primary'], edgecolor='white', linewidth=0.5,
                   yerr=np.array(oracle_cis).T, capsize=4, error_kw={'linewidth': 1})
    bars2 = ax.bar(x + width/2, realistic_means, width, label='Realistic',
                   color=COLORS['secondary'], edgecolor='white', linewidth=0.5,
                   yerr=np.array(realistic_cis).T, capsize=4, error_kw={'linewidth': 1})

    # Annotate gaps
    for i, gap in enumerate(gaps):
        mid_y = (oracle_means[i] + realistic_means[i]) / 2
        ax.annotate(f'{gap:.0f}% gap',
                    xy=(x[i] + 0.02, mid_y),
                    fontsize=8, fontweight='bold', color=COLORS['gray'],
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=COLORS['gray'], alpha=0.8, linewidth=0.5))

    ax.set_xlabel('Dataset')
    ax.set_ylabel('NDCG@10')
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 0.14)

    fig.savefig(OUTPUT / 'fig_oracle_realistic_gap.pdf')
    fig.savefig(OUTPUT / 'fig_oracle_realistic_gap.png')
    print(f"Figure 2 saved: {OUTPUT / 'fig_oracle_realistic_gap.pdf'}")
    plt.close(fig)


# ============================================================================
# Figure 3: Model Scaling Failure
# ============================================================================
def figure3_model_scaling():
    """Scatter plot showing no improvement with model scale."""
    beauty = load_json('kdd_multi_model_unified.json')
    movies = load_json('kdd_multi_model_movies.json')

    # Map model names to parameter counts (numeric for x-axis)
    param_map = {
        'Gemma3-4B': 4,
        'Llama3.1-8B': 8,
        'GPT-4o-mini': 8,  # estimated
        'Qwen3-Next-80B': 80,
        'DeepSeek-V3.2': 671,
        'Mistral-Large-675B': 675,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0), sharey=True)

    # ---- Beauty subplot ----
    cf_beauty = beauty['results']['CF_Score']['ndcg_mean']
    for name, data in beauty['results'].items():
        if name in ('Random', 'CF_Score'):
            continue
        params = param_map.get(name, None)
        if params is None:
            continue
        mean = data['ndcg_mean']
        ci_lo = data['ndcg_ci_low']
        ci_hi = data['ndcg_ci_high']
        ax1.errorbar(params, mean, yerr=[[mean - ci_lo], [ci_hi - mean]],
                     fmt='o', markersize=7, color=COLORS['primary'],
                     ecolor=COLORS['primary'], elinewidth=1, capsize=3,
                     markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    ax1.axhline(y=cf_beauty, color=COLORS['cf_baseline'], linestyle='--',
                linewidth=1.2, label=f'CF baseline ({cf_beauty:.4f})', zorder=3)
    ax1.set_xscale('log')
    ax1.set_xlabel('Parameters (B)')
    ax1.set_ylabel('NDCG@10')
    ax1.set_title('Beauty (Recall 8.1%)', fontsize=10)
    ax1.set_xlim(2, 1000)
    ax1.set_ylim(-0.002, 0.025)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{int(x)}B'))

    # ---- Movies subplot ----
    cf_movies = movies['results']['CF_Score']['ndcg_mean']
    for name, data in movies['results'].items():
        if name in ('Random', 'CF_Score'):
            continue
        # Parse params
        params_str = data.get('params', '')
        if '671B' in str(params_str):
            params = 671
        elif '675B' in str(params_str):
            params = 675
        elif '80B' in str(params_str):
            params = 80
        elif '8B' in str(params_str):
            params = 8
        else:
            continue
        mean = data['ndcg_mean']
        ci_lo = data['ndcg_ci_low']
        ci_hi = data['ndcg_ci_high']
        ax2.errorbar(params, mean, yerr=[[mean - ci_lo], [ci_hi - mean]],
                     fmt='s', markersize=7, color=COLORS['secondary'],
                     ecolor=COLORS['secondary'], elinewidth=1, capsize=3,
                     markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    ax2.axhline(y=cf_movies, color=COLORS['cf_baseline'], linestyle='--',
                linewidth=1.2, label=f'CF baseline ({cf_movies:.4f})', zorder=3)
    ax2.set_xscale('log')
    ax2.set_xlabel('Parameters (B)')
    ax2.set_title('Movies (Recall 3.2%)', fontsize=10)
    ax2.set_xlim(5, 1000)
    ax2.legend(fontsize=7, loc='upper left')
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{int(x)}B'))

    plt.tight_layout()
    fig.savefig(OUTPUT / 'fig_model_scaling.pdf')
    fig.savefig(OUTPUT / 'fig_model_scaling.png')
    print(f"Figure 3 saved: {OUTPUT / 'fig_model_scaling.pdf'}")
    plt.close(fig)


# ============================================================================
# Figure 4: Recall-Performance Relationship
# ============================================================================
def figure4_recall_performance():
    """Line plot showing NDCG scales linearly with recall."""
    rs = load_json('kdd_recall_sensitivity.json')

    recall_levels = []
    p1_means, p1_ci_lo, p1_ci_hi = [], [], []
    p3_means, p3_ci_lo, p3_ci_hi = [], [], []

    for recall_str in ['0.05', '0.1', '0.2', '0.5', '1.0']:
        data = rs['results'][recall_str]
        recall_levels.append(float(recall_str) * 100)
        p1_means.append(data['P1']['mean'])
        p1_ci_lo.append(data['P1']['ci_low'])
        p1_ci_hi.append(data['P1']['ci_high'])
        p3_means.append(data['P3']['mean'])
        p3_ci_lo.append(data['P3']['ci_low'])
        p3_ci_hi.append(data['P3']['ci_high'])

    recall_levels = np.array(recall_levels)
    p1_means = np.array(p1_means)
    p1_ci_lo = np.array(p1_ci_lo)
    p1_ci_hi = np.array(p1_ci_hi)
    p3_means = np.array(p3_means)
    p3_ci_lo = np.array(p3_ci_lo)
    p3_ci_hi = np.array(p3_ci_hi)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.plot(recall_levels, p1_means, 'o-', color=COLORS['primary'],
            label='P1 (Basic)', markersize=6, linewidth=1.5, zorder=5)
    ax.fill_between(recall_levels, p1_ci_lo, p1_ci_hi,
                    alpha=0.15, color=COLORS['primary'])

    ax.plot(recall_levels, p3_means, 's--', color=COLORS['secondary'],
            label='P3 (Enhanced CoT)', markersize=6, linewidth=1.5, zorder=5)
    ax.fill_between(recall_levels, p3_ci_lo, p3_ci_hi,
                    alpha=0.15, color=COLORS['secondary'])

    # Annotate realistic recall range
    ax.axvspan(2, 9, alpha=0.1, color=COLORS['gray'], zorder=0)
    ax.annotate('Realistic recall\nrange (2--9%)', xy=(5.5, 0.001),
                fontsize=7, color=COLORS['gray'], ha='center', style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.8))

    ax.set_xlabel('Recall (%)')
    ax.set_ylabel('NDCG@10')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 0.075)

    fig.savefig(OUTPUT / 'fig_recall_performance.pdf')
    fig.savefig(OUTPUT / 'fig_recall_performance.png')
    print(f"Figure 4 saved: {OUTPUT / 'fig_recall_performance.pdf'}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Generating publication figures...")
    figure1_catalog_vs_recall()
    figure2_oracle_realistic_gap()
    figure3_model_scaling()
    figure4_recall_performance()
    print("\nAll figures generated successfully!")
