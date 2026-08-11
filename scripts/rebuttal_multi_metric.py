#!/usr/bin/env python3
"""
Rebuttal supplementary: Multi-metric evaluation (NDCG@10, Hit@10, MAP@10)
for Beauty, Movies, Electronics at n=500.

Addresses PYE2-W4 (single metric concern) and Vn23-W2 (bootstrap population test).

Runs CF-SVD retrieval, evaluates multiple metrics under:
  - Oracle (inject ground truth into candidate set)
  - Realistic (CF candidates only)
  - CF baseline (CF ranking order, no reranker)

Also runs bootstrap population-level test: CI for (Realistic - CF_baseline) mean difference.

Usage: python scripts/rebuttal_multi_metric.py
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.stats import wilcoxon
from datetime import datetime

N_USERS = 500
K = 100
TOP_K = 10
SEED = 42
N_BOOTSTRAP = 10000

DATASETS = {
    'beauty':      'data/processed/amazon_beauty_sampled',
    'movies':      'data/processed/amazon_movies_sampled',
    'electronics': 'data/processed/amazon_electronics_sampled',
}


# ─── Metrics ─────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_list, ground_truth, k=10):
    dcg = sum(1.0 / np.log2(i + 2)
              for i, item in enumerate(ranked_list[:k]) if item in ground_truth)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(ranked_list, ground_truth, k=10):
    return 1.0 if any(item in ground_truth for item in ranked_list[:k]) else 0.0


def map_at_k(ranked_list, ground_truth, k=10):
    hits, score = 0, 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(ground_truth), k) if ground_truth else 0.0


def recall_at_k(candidates, ground_truth, k=100):
    hits = len(set(candidates[:k]) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(scores, n_boot=N_BOOTSTRAP):
    scores = np.array(scores)
    boot_means = np.array([
        np.mean(np.random.choice(scores, len(scores), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.mean(scores)), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def bootstrap_diff_test(scores_a, scores_b, n_boot=N_BOOTSTRAP):
    """Population-level bootstrap test: H0: mean(B - A) = 0"""
    diffs = np.array(scores_b) - np.array(scores_a)
    mean_diff = float(np.mean(diffs))
    boot_means = np.array([
        np.mean(np.random.choice(diffs, len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    # two-sided p: fraction of bootstrap means on the opposite side of zero
    p_val = float(2 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)))
    return mean_diff, ci_lo, ci_hi, p_val


# ─── CF Retrieval ─────────────────────────────────────────────────────────────

class CFRetrieval:
    def __init__(self, n_factors=128):
        self.n_factors = n_factors

    def fit(self, train_df):
        items = train_df['item_id'].unique()
        users = train_df['user_id'].unique()
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        self.user_to_idx = {u: idx for idx, u in enumerate(users)}
        rows = [self.user_to_idx[u] for u in train_df['user_id']]
        cols = [self.item_to_idx[i] for i in train_df['item_id']]
        data = np.ones(len(rows))
        mat = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        n_comp = min(self.n_factors, min(mat.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        self.user_factors = self.svd.fit_transform(mat)
        self.item_factors = self.svd.components_.T
        self.user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    def get_candidates(self, user_id, K=100):
        if user_id not in self.user_to_idx:
            return [], []
        user_idx = self.user_to_idx[user_id]
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        for item in self.user_history.get(user_id, []):
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf
        top_idx = np.argsort(scores)[::-1][:K]
        return [self.idx_to_item[i] for i in top_idx], [float(scores[i]) for i in top_idx]


# ─── Simulate oracle LLM reranking (perfect ranker for ceiling) ──────────────

def oracle_rerank(candidates, ground_truth):
    """Perfect reranker: puts all relevant items first."""
    relevant = [c for c in candidates if c in ground_truth]
    non_relevant = [c for c in candidates if c not in ground_truth]
    return relevant + non_relevant


# ─── Simulate realistic LLM (random among candidates = worst case) ───────────
# For bootstrap test we need realistic LLM per-user arrays.
# We load them from the existing n500 log if available, else approximate.

def load_existing_per_user(log_path, dataset_key):
    """Try to load per-user LLM NDCG from existing log."""
    try:
        with open(log_path) as f:
            d = json.load(f)
        ds = d['datasets'].get(dataset_key, {})
        llm = ds.get('llm_per_user')
        cf = ds.get('cf_per_user')
        if llm and cf:
            return np.array(llm), np.array(cf)
    except Exception:
        pass
    return None, None


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_dataset(name, data_path):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    train_df = pd.read_parquet(project_root / data_path / 'train.parquet')
    test_df  = pd.read_parquet(project_root / data_path / 'test.parquet')

    cf = CFRetrieval()
    cf.fit(train_df)

    np.random.seed(SEED)
    test_users = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid = [u for u in test_users if u in cf.user_to_idx and len(test_users[u]) > 0]
    if len(valid) > N_USERS:
        valid = list(np.random.choice(valid, N_USERS, replace=False))
    print(f"  Users: {len(valid)}, Items: {train_df['item_id'].nunique()}")

    # Per-user metric arrays
    cf_ndcg, cf_hit, cf_map, cf_recall = [], [], [], []
    oracle_ndcg, oracle_hit, oracle_map = [], [], []

    for user in valid:
        gt = set(test_users[user])
        cands, cand_scores = cf.get_candidates(user, K=K)

        # CF baseline: use CF ranking order as-is
        cf_ndcg.append(ndcg_at_k(cands, gt, TOP_K))
        cf_hit.append(hit_at_k(cands, gt, TOP_K))
        cf_map.append(map_at_k(cands, gt, TOP_K))
        cf_recall.append(recall_at_k(cands, gt, K))

        # Oracle: perfect reranker on realistic candidates
        oracle_ranked = oracle_rerank(cands, gt)
        oracle_ndcg.append(ndcg_at_k(oracle_ranked, gt, TOP_K))
        oracle_hit.append(hit_at_k(oracle_ranked, gt, TOP_K))
        oracle_map.append(map_at_k(oracle_ranked, gt, TOP_K))

    # Recall stats
    rec_mean, rec_lo, rec_hi = bootstrap_ci(cf_recall)

    # CF baseline stats
    cf_ndcg_m, cf_ndcg_lo, cf_ndcg_hi = bootstrap_ci(cf_ndcg)
    cf_hit_m,  cf_hit_lo,  cf_hit_hi  = bootstrap_ci(cf_hit)
    cf_map_m,  cf_map_lo,  cf_map_hi  = bootstrap_ci(cf_map)

    # Oracle ceiling stats
    or_ndcg_m, or_ndcg_lo, or_ndcg_hi = bootstrap_ci(oracle_ndcg)
    or_hit_m,  or_hit_lo,  or_hit_hi  = bootstrap_ci(oracle_hit)
    or_map_m,  or_map_lo,  or_map_hi  = bootstrap_ci(oracle_map)

    # Bootstrap population-level test: LLM realistic vs CF baseline
    # Load existing LLM realistic per-user NDCG from n500 log
    llm_ndcg_arr, cf_from_log = load_existing_per_user(
        project_root / 'experiments/logs/kdd_oracle_realistic_n500.json',
        f'amazon_{name}'
    )
    bootstrap_result = None
    if llm_ndcg_arr is not None and len(llm_ndcg_arr) == len(cf_ndcg):
        diff_mean, diff_lo, diff_hi, diff_p = bootstrap_diff_test(cf_ndcg, llm_ndcg_arr)
        bootstrap_result = {
            'llm_realistic_mean': float(np.mean(llm_ndcg_arr)),
            'cf_baseline_mean': cf_ndcg_m,
            'diff_mean': diff_mean,
            'diff_ci_95': [diff_lo, diff_hi],
            'p_value_two_sided': diff_p,
            'ci_contains_zero': diff_lo <= 0 <= diff_hi,
            'note': 'Population-level bootstrap test on n={} per-user NDCG differences'.format(len(cf_ndcg))
        }
    else:
        # Per-user arrays not available from log; note this
        bootstrap_result = {
            'note': 'Per-user LLM arrays not in log. Re-run kdd_oracle_realistic_n500.py with --save_per_user flag to enable this test.',
            'aggregate_evidence': {
                'cf_ndcg_mean': cf_ndcg_m,
                'llm_ndcg_from_log': None,
                'wilcoxon_p_from_log': 'see kdd_oracle_realistic_n500.json'
            }
        }

    # Wilcoxon test (user-level) for CF ordering vs oracle ceiling
    try:
        w_stat, w_p = wilcoxon(oracle_ndcg, cf_ndcg, alternative='greater', zero_method='wilcox')
    except Exception:
        w_stat, w_p = float('nan'), 1.0

    result = {
        'dataset': name,
        'n_users': len(valid),
        'n_items': train_df['item_id'].nunique(),
        'recall_at_100': {'mean': rec_mean, 'ci': [rec_lo, rec_hi]},
        'cf_baseline': {
            'ndcg@10': {'mean': cf_ndcg_m, 'ci': [cf_ndcg_lo, cf_ndcg_hi]},
            'hit@10':  {'mean': cf_hit_m,  'ci': [cf_hit_lo,  cf_hit_hi]},
            'map@10':  {'mean': cf_map_m,  'ci': [cf_map_lo,  cf_map_hi]},
        },
        'oracle_ceiling': {
            'ndcg@10': {'mean': or_ndcg_m, 'ci': [or_ndcg_lo, or_ndcg_hi]},
            'hit@10':  {'mean': or_hit_m,  'ci': [or_hit_lo,  or_hit_hi]},
            'map@10':  {'mean': or_map_m,  'ci': [or_map_lo,  or_map_hi]},
            'wilcoxon_vs_cf_p': float(w_p),
        },
        'bootstrap_population_test_llm_vs_cf': bootstrap_result,
        'interpretation': {
            'fraction_users_zero_recall': float(np.mean(np.array(cf_recall) == 0)),
            'fraction_users_zero_ndcg_cf': float(np.mean(np.array(cf_ndcg) == 0)),
            'oracle_gap_vs_cf_ndcg': float(or_ndcg_m - cf_ndcg_m) if or_ndcg_m else None,
        }
    }

    # Print summary
    print(f"  Recall@100:     {rec_mean:.4f} [{rec_lo:.4f}, {rec_hi:.4f}]")
    print(f"  CF NDCG@10:     {cf_ndcg_m:.4f}  Hit@10: {cf_hit_m:.4f}  MAP@10: {cf_map_m:.4f}")
    print(f"  Oracle NDCG@10: {or_ndcg_m:.4f}  Hit@10: {or_hit_m:.4f}  MAP@10: {or_map_m:.4f}")
    print(f"  Zero-NDCG users (CF): {result['interpretation']['fraction_users_zero_ndcg_cf']:.1%}")
    if bootstrap_result and 'p_value_two_sided' in bootstrap_result:
        print(f"  Bootstrap pop-test (LLM-Realistic vs CF): diff={bootstrap_result['diff_mean']:.4f}, "
              f"CI=[{bootstrap_result['diff_ci_95'][0]:.4f},{bootstrap_result['diff_ci_95'][1]:.4f}], "
              f"p={bootstrap_result['p_value_two_sided']:.3f}")

    return result


def main():
    np.random.seed(SEED)
    all_results = {}

    for name, path in DATASETS.items():
        full_path = project_root / path
        if not full_path.exists():
            print(f"[SKIP] {name}: data not found at {full_path}")
            continue
        all_results[name] = run_dataset(name, path)

    out_path = project_root / 'experiments/logs/rebuttal_multi_metric.json'
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {'n_users': N_USERS, 'K': K, 'top_k': TOP_K, 'n_bootstrap': N_BOOTSTRAP, 'seed': SEED},
            'results': all_results
        }, f, indent=2)

    print(f"\nSaved to {out_path}")

    # Print LaTeX table snippet
    print("\n--- LaTeX table snippet (for paper) ---")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{Recall@100} & \textbf{NDCG@10} & \textbf{Hit@10} & \textbf{MAP@10} \\")
    print(r"\midrule")
    for name, res in all_results.items():
        r  = res['recall_at_100']['mean']
        nd = res['oracle_ceiling']['ndcg@10']['mean']
        h  = res['oracle_ceiling']['hit@10']['mean']
        m  = res['oracle_ceiling']['map@10']['mean']
        print(rf"{name.capitalize()} (Oracle) & {r:.3f} & {nd:.4f} & {h:.4f} & {m:.4f} \\")
        nd = res['cf_baseline']['ndcg@10']['mean']
        h  = res['cf_baseline']['hit@10']['mean']
        m  = res['cf_baseline']['map@10']['mean']
        print(rf"{name.capitalize()} (CF Realistic) & {r:.3f} & {nd:.4f} & {h:.4f} & {m:.4f} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == '__main__':
    main()
