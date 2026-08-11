#!/usr/bin/env python3
"""
KDD 2026 Supplementary Experiment: Statistical Significance Testing
====================================================================
Provides rigorous statistical analysis of experimental results including:
- Paired bootstrap test
- Wilcoxon signed-rank test
- Holm-Bonferroni correction for multiple comparisons

Usage:
    python scripts/kdd_statistical_significance.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def paired_bootstrap_test(scores_a: List[float], scores_b: List[float],
                          n_bootstrap: int = 10000) -> Dict:
    """
    Paired bootstrap test for comparing two methods on the same users.

    Returns:
        dict with mean_diff, ci_lower, ci_upper, p_value
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)

    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # Bootstrap differences
    boot_diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        boot_diff = np.mean(scores_a[indices]) - np.mean(scores_b[indices])
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)

    # Confidence interval
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    # Two-tailed p-value (proportion of bootstrap samples with opposite sign)
    if observed_diff >= 0:
        p_value = 2 * np.mean(boot_diffs <= 0)
    else:
        p_value = 2 * np.mean(boot_diffs >= 0)

    return {
        'mean_diff': observed_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': min(p_value, 1.0),
        'significant_at_0.05': p_value < 0.05
    }


def wilcoxon_test(scores_a: List[float], scores_b: List[float]) -> Dict:
    """
    Wilcoxon signed-rank test for paired samples.

    Returns:
        dict with statistic, p_value, significant_at_0.05
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    # Remove ties (where difference is exactly 0)
    diff = scores_a - scores_b
    non_zero_mask = diff != 0

    if np.sum(non_zero_mask) < 10:
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant_at_0.05': False,
            'note': 'Too few non-zero differences'
        }

    try:
        statistic, p_value = stats.wilcoxon(scores_a, scores_b, alternative='two-sided')
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05
        }
    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant_at_0.05': False,
            'note': str(e)
        }


def holm_bonferroni_correction(p_values: Dict[str, float], alpha: float = 0.05) -> Dict:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: dict mapping comparison name to p-value
        alpha: family-wise error rate

    Returns:
        dict with corrected significance decisions
    """
    comparisons = sorted(p_values.items(), key=lambda x: x[1])
    n = len(comparisons)

    results = {}
    for i, (name, p) in enumerate(comparisons):
        adjusted_alpha = alpha / (n - i)
        significant = p <= adjusted_alpha
        results[name] = {
            'original_p': p,
            'adjusted_alpha': adjusted_alpha,
            'rank': i + 1,
            'significant_after_correction': significant
        }

    return results


def analyze_results():
    """Analyze existing experimental results for statistical significance."""

    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)

    # Load comprehensive results
    results_path = project_root / 'experiments' / 'logs' / 'kdd_comprehensive_results.json'

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)

    beauty_data = data.get('beauty', {})
    reranking = beauty_data.get('reranking', {})

    if not reranking:
        print("No reranking results found.")
        return

    # Extract per-user scores for each method
    methods = list(reranking.keys())
    print(f"\nMethods: {methods}")

    # Pairwise comparisons
    print("\n" + "=" * 70)
    print("PAIRWISE COMPARISONS")
    print("=" * 70)

    comparisons = {}
    p_values = {}

    for i, method_a in enumerate(methods):
        for method_b in methods[i+1:]:
            scores_a = reranking[method_a].get('ndcg@10', {}).get('scores', [])
            scores_b = reranking[method_b].get('ndcg@10', {}).get('scores', [])

            if not scores_a or not scores_b:
                continue

            comparison_name = f"{method_a} vs {method_b}"
            print(f"\n{comparison_name}")
            print("-" * 40)

            # Paired bootstrap test
            boot_result = paired_bootstrap_test(scores_a, scores_b)
            print(f"  Paired Bootstrap Test:")
            print(f"    Mean difference: {boot_result['mean_diff']:.6f}")
            print(f"    95% CI: [{boot_result['ci_lower']:.6f}, {boot_result['ci_upper']:.6f}]")
            print(f"    p-value: {boot_result['p_value']:.4f}")
            print(f"    Significant (α=0.05): {boot_result['significant_at_0.05']}")

            # Wilcoxon test
            wilcox_result = wilcoxon_test(scores_a, scores_b)
            print(f"  Wilcoxon Signed-Rank Test:")
            print(f"    Statistic: {wilcox_result['statistic']:.2f}" if not np.isnan(wilcox_result['statistic']) else "    Statistic: N/A")
            print(f"    p-value: {wilcox_result['p_value']:.4f}")
            print(f"    Significant (α=0.05): {wilcox_result['significant_at_0.05']}")

            comparisons[comparison_name] = {
                'bootstrap': boot_result,
                'wilcoxon': wilcox_result
            }
            p_values[comparison_name] = boot_result['p_value']

    # Holm-Bonferroni correction
    if p_values:
        print("\n" + "=" * 70)
        print("HOLM-BONFERRONI CORRECTION")
        print("=" * 70)

        corrected = holm_bonferroni_correction(p_values)

        print(f"\nNumber of comparisons: {len(p_values)}")
        print(f"Family-wise error rate: α = 0.05\n")

        print(f"{'Comparison':<30} {'p-value':<10} {'Adj. α':<10} {'Rank':<6} {'Significant'}")
        print("-" * 70)

        for name, result in sorted(corrected.items(), key=lambda x: x[1]['rank']):
            sig = "YES" if result['significant_after_correction'] else "NO"
            print(f"{name:<30} {result['original_p']:<10.4f} {result['adjusted_alpha']:<10.4f} {result['rank']:<6} {sig}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for method in methods:
        scores = reranking[method].get('ndcg@10', {}).get('scores', [])
        if scores:
            mean = np.mean(scores)
            std = np.std(scores)
            median = np.median(scores)
            iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
            non_zero = np.sum(np.array(scores) > 0) / len(scores) * 100

            print(f"\n{method}:")
            print(f"  Mean ± Std: {mean:.4f} ± {std:.4f}")
            print(f"  Median (IQR): {median:.4f} ({iqr:.4f})")
            print(f"  Non-zero rate: {non_zero:.1f}%")
            print(f"  N samples: {len(scores)}")

    # Effect size analysis
    print("\n" + "=" * 70)
    print("EFFECT SIZE ANALYSIS (Cohen's d)")
    print("=" * 70)

    for i, method_a in enumerate(methods):
        for method_b in methods[i+1:]:
            scores_a = np.array(reranking[method_a].get('ndcg@10', {}).get('scores', []))
            scores_b = np.array(reranking[method_b].get('ndcg@10', {}).get('scores', []))

            if len(scores_a) == 0 or len(scores_b) == 0:
                continue

            # Cohen's d for paired samples
            diff = scores_a - scores_b
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

            effect_size = "negligible" if abs(cohens_d) < 0.2 else \
                         "small" if abs(cohens_d) < 0.5 else \
                         "medium" if abs(cohens_d) < 0.8 else "large"

            print(f"  {method_a} vs {method_b}: d = {cohens_d:.3f} ({effect_size})")

    # Save results
    output = {
        'comparisons': comparisons,
        'holm_bonferroni': corrected if p_values else {},
        'summary': {
            method: {
                'mean': float(np.mean(reranking[method].get('ndcg@10', {}).get('scores', [0]))),
                'std': float(np.std(reranking[method].get('ndcg@10', {}).get('scores', [0]))),
                'n': len(reranking[method].get('ndcg@10', {}).get('scores', []))
            }
            for method in methods
        }
    }

    output_path = project_root / 'experiments' / 'logs' / 'kdd_statistical_significance.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\nResults saved to: {output_path}")

    return output


def analyze_retrieval_results():
    """Analyze retrieval baseline results for statistical significance."""

    print("\n" + "=" * 70)
    print("RETRIEVAL BASELINES STATISTICAL ANALYSIS")
    print("=" * 70)

    results_path = project_root / 'experiments' / 'logs' / 'kdd_retrieval_baselines.json'

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)

    for dataset in ['beauty', 'movies', 'electronics']:
        if dataset not in data:
            continue

        print(f"\n=== {dataset.upper()} ===")

        results = data[dataset]['results']
        methods = list(results.keys())

        # Summary
        print(f"\n{'Method':<12} {'Mean':<10} {'95% CI':<20} {'Non-zero %'}")
        print("-" * 55)

        for method in methods:
            scores = results[method].get('scores', [])
            if scores:
                mean = np.mean(scores)
                ci_low = results[method].get('ci_low', 0)
                ci_high = results[method].get('ci_high', 0)
                non_zero = np.sum(np.array(scores) > 0) / len(scores) * 100
                print(f"{method:<12} {mean:.4f}     [{ci_low:.4f}, {ci_high:.4f}]  {non_zero:.1f}%")


if __name__ == '__main__':
    analyze_results()
    analyze_retrieval_results()
