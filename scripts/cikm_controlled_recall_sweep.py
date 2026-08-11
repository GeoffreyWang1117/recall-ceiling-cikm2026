#!/usr/bin/env python3
"""
CIKM 2026 P0-A: Controlled Recall Sweep at Fixed Catalog Size
==============================================================

Defends the "8.8% reranker-significance boundary" against the round-1
critique that it is established from one datapoint per side with confounded
catalog sizes (Beauty 1.4K @ 8.2% n.s. vs Toys 12K @ 8.8% p<.0001).

Design
------
Two arms, both on Amazon Toys (catalog 9,545 items, 1,594 users):

  Arm 1 — FULL CATALOG (~9.5K items):
    Sweep K in {20, 50, 100, 150, 200, 300}, retrieve CF candidates,
    score with LambdaMART trained on Toys train split. Naturally
    produces recall in roughly {2%, 5%, 8.8%, 12%, 14%, 18%}.

  Arm 2 — SUBSAMPLED CATALOG (~1.5K items, Beauty-equivalent):
    Sub-sample item universe to 1,594 items (matches Beauty), rebuild
    CF and LambdaMART on the reduced data, repeat the same K sweep.

For each (arm, K) we report:
    real_recall, NDCG_CF, NDCG_LambdaMART, Wilcoxon p (LambdaMART vs CF)

Decision rule (will be reported in paper §4.4):
    If LambdaMART significance transitions smoothly with recall on BOTH
    arms -> recall drives the boundary, the 8.8% number is a descriptor.
    If transition disappears under arm 2 -> catalog is the confound,
    the numeric boundary must be retracted.

Usage
-----
    python scripts/cikm_controlled_recall_sweep.py --n_users 500
    python scripts/cikm_controlled_recall_sweep.py --n_users 500 --quick   # smoke test

Output
------
    experiments/logs/cikm_controlled_recall_sweep.json
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import wilcoxon
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm


# ============================================================================
# CF Retrieval (same SVD config used throughout the paper)
# ============================================================================

class CFSVD:
    def __init__(self, n_factors=128):
        self.n_factors = n_factors

    def fit(self, train_df):
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}

        rows = [self.user_to_idx[u] for u in train_df['user_id']]
        cols = [self.item_to_idx[i] for i in train_df['item_id']]
        data = np.ones(len(rows))
        ui = csr_matrix((data, (rows, cols)),
                        shape=(len(users), len(items)))

        n_components = min(self.n_factors, min(ui.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = svd.fit_transform(ui)
        self.item_factors = svd.components_.T
        self.popularity = train_df['item_id'].value_counts().to_dict()

    def topk(self, user_id, K, exclude_ids):
        if user_id not in self.user_to_idx:
            return [], []
        u = self.user_factors[self.user_to_idx[user_id]]
        scores = self.item_factors @ u
        for it in exclude_ids:
            if it in self.item_to_idx:
                scores[self.item_to_idx[it]] = -np.inf
        top = np.argsort(scores)[::-1][:K]
        cands = [self.idx_to_item[i] for i in top if scores[i] > -np.inf]
        s = [scores[self.item_to_idx[c]] for c in cands]
        if s:
            lo, hi = min(s), max(s)
            if hi > lo:
                s = [(x - lo) / (hi - lo) for x in s]
        return cands, s

    def user_emb(self, uid):
        if uid in self.user_to_idx:
            return self.user_factors[self.user_to_idx[uid]]
        return np.zeros(self.user_factors.shape[1])

    def item_emb(self, iid):
        if iid in self.item_to_idx:
            return self.item_factors[self.item_to_idx[iid]]
        return np.zeros(self.item_factors.shape[1])


# ============================================================================
# Feature engineering — identical to kdd_neural_reranker_baselines.py
# (kept inline so this script is self-contained and the tracker can audit it)
# ============================================================================

def build_features(uid, cands, cf_scores, cf, user_history, item_pop):
    feats = []
    u_emb = cf.user_emb(uid)
    u_norm = np.linalg.norm(u_emb) + 1e-8
    history = user_history.get(uid, [])
    last10 = history[-10:]

    for i, (item, score) in enumerate(zip(cands, cf_scores)):
        i_emb = cf.item_emb(item)
        sim = float(np.dot(u_emb, i_emb) / (u_norm * (np.linalg.norm(i_emb) + 1e-8)))
        pop = np.log1p(item_pop.get(item, 0))

        cooc = 0.0
        if item in cf.item_to_idx:
            ii = cf.item_to_idx[item]
            v = cf.item_factors[ii]
            for h in last10:
                if h in cf.item_to_idx:
                    cooc += float(np.dot(v, cf.item_factors[cf.item_to_idx[h]]))
            cooc /= max(len(last10), 1)

        feats.append([
            score,
            i / max(len(cands), 1),
            pop,
            sim,
            np.log1p(len(history)),
            cooc,
            1.0 / (i + 1),
            float(np.linalg.norm(i_emb)),
        ])
    return np.array(feats, dtype=np.float32)


# ============================================================================
# LambdaMART (same lightgbm config as kdd_neural_reranker_baselines.py)
# ============================================================================

class LambdaMART:
    def __init__(self, n_leaves=31, n_estimators=100, lr=0.1):
        self.params = dict(
            objective='lambdarank', metric='ndcg',
            ndcg_eval_at=[10],
            num_leaves=n_leaves, learning_rate=lr,
            n_estimators=n_estimators, verbose=-1, seed=42,
        )
        self.model = None

    def fit(self, X, y, groups):
        import lightgbm as lgb
        ds = lgb.Dataset(X, label=y, group=groups)
        self.model = lgb.train(self.params, ds,
                               num_boost_round=self.params['n_estimators'])

    def predict(self, X):
        return self.model.predict(X)


# ============================================================================
# Evaluation
# ============================================================================

def ndcg_at_k(ranked, gt, k=10):
    gt_set = set(gt) if isinstance(gt, list) else {gt}
    dcg = 0.0
    for i, it in enumerate(ranked[:k]):
        if it in gt_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def safe_wilcoxon(a, b):
    """Wilcoxon signed-rank, robust to all-zero diffs."""
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    if np.allclose(diff, 0) or np.count_nonzero(diff) < 5:
        return float('nan')
    try:
        _, p = wilcoxon(a, b)
        return float(p)
    except Exception:
        return float('nan')


def bootstrap_ci(scores, n=1000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    s = np.asarray(scores)
    if len(s) == 0:
        return 0.0, 0.0, 0.0
    boots = rng.choice(s, size=(n, len(s)), replace=True).mean(axis=1)
    a = (1 - ci) / 2
    return float(s.mean()), float(np.percentile(boots, a * 100)), float(np.percentile(boots, (1 - a) * 100))


# ============================================================================
# Catalog subsampling — keep only `keep_n` most popular items
# Rationale: random subsample creates a recall floor of zero for many users;
# popularity-based subsample preserves enough density to make CF trainable
# and matches how Beauty was originally curated (popularity-truncated).
# ============================================================================

def subsample_catalog(train_df, test_df, keep_n, seed=42):
    item_pop = train_df['item_id'].value_counts()
    keep_items = set(item_pop.head(keep_n).index)
    new_train = train_df[train_df['item_id'].isin(keep_items)].copy()
    new_test = test_df[test_df['item_id'].isin(keep_items)].copy()
    # Drop users with no remaining train interactions
    keep_users = set(new_train['user_id'].unique())
    new_test = new_test[new_test['user_id'].isin(keep_users)]
    return new_train, new_test


# ============================================================================
# Single-arm experiment
# ============================================================================

def run_arm(arm_name, train_df, test_df, K_values, n_users, seed=42):
    print(f"\n{'='*72}\nARM: {arm_name}\n{'='*72}")
    print(f"  Catalog size: {train_df['item_id'].nunique()}")
    print(f"  Train interactions: {len(train_df)}")
    print(f"  Test users: {test_df['user_id'].nunique()}")

    np.random.seed(seed)

    # Build CF
    cf = CFSVD(n_factors=128)
    cf.fit(train_df)
    item_pop = train_df['item_id'].value_counts().to_dict()
    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()

    valid_users = [u for u in test_gt
                   if u in user_history and len(user_history[u]) >= 3]
    if len(valid_users) > n_users:
        valid_users = list(np.random.choice(valid_users, size=n_users, replace=False))
    print(f"  Eval users: {len(valid_users)}")

    # ------------------------------------------------------------------
    # Train LambdaMART once, on the LARGEST K we will evaluate
    # so the training distribution covers all the smaller K subsets.
    # Use a held-out half of valid_users for training features.
    # ------------------------------------------------------------------
    K_max = max(K_values)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(valid_users)
    split = max(50, len(perm) // 2)
    train_users = list(perm[:split])
    eval_users = list(perm[split:]) if len(perm) > split else list(perm)
    print(f"  LambdaMART train users: {len(train_users)}, eval users: {len(eval_users)}")

    print("  Building LambdaMART training features...")
    Xs, ys, groups = [], [], []
    for uid in tqdm(train_users, desc='    train feats'):
        cands, scores = cf.topk(uid, K_max, user_history.get(uid, []))
        if len(cands) == 0:
            continue
        feats = build_features(uid, cands, scores, cf, user_history, item_pop)
        gt = set(test_gt.get(uid, []))
        labels = np.array([1 if c in gt else 0 for c in cands], dtype=np.int32)
        Xs.append(feats)
        ys.append(labels)
        groups.append(len(cands))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    print(f"    Total training rows: {len(X)}, positive rate: {y.mean():.4f}")
    lm = LambdaMART()
    lm.fit(X, y, groups)

    # ------------------------------------------------------------------
    # K sweep on eval_users
    # ------------------------------------------------------------------
    rows = []
    for K in K_values:
        cf_ndcgs, lm_ndcgs, recalls = [], [], []
        for uid in eval_users:
            cands, scores = cf.topk(uid, K, user_history.get(uid, []))
            if len(cands) == 0:
                continue
            gt = test_gt.get(uid, [])
            gt_set = set(gt)
            recalls.append(len(set(cands) & gt_set) / max(len(gt_set), 1))

            cf_ndcgs.append(ndcg_at_k(cands, gt, k=10))
            feats = build_features(uid, cands, scores, cf, user_history, item_pop)
            preds = lm.predict(feats)
            order = np.argsort(preds)[::-1]
            reranked = [cands[i] for i in order]
            lm_ndcgs.append(ndcg_at_k(reranked, gt, k=10))

        cf_mean, cf_lo, cf_hi = bootstrap_ci(cf_ndcgs, seed=seed)
        lm_mean, lm_lo, lm_hi = bootstrap_ci(lm_ndcgs, seed=seed + 1)
        p = safe_wilcoxon(lm_ndcgs, cf_ndcgs)
        r_mean = float(np.mean(recalls)) if recalls else 0.0

        # ceiling utilisation: NDCG / theoretical max given recall (LOO -> recall directly bounds NDCG@10)
        eta_lm = lm_mean / r_mean if r_mean > 0 else 0.0

        row = dict(
            K=K, n_eval=len(cf_ndcgs),
            recall_mean=r_mean,
            cf_ndcg=cf_mean, cf_ci=[cf_lo, cf_hi],
            lm_ndcg=lm_mean, lm_ci=[lm_lo, lm_hi],
            wilcoxon_p=p,
            eta_lambdamart=eta_lm,
            delta=lm_mean - cf_mean,
        )
        rows.append(row)
        print(f"  K={K:>3} | recall={r_mean*100:5.2f}% | "
              f"CF={cf_mean:.4f} | LambdaMART={lm_mean:.4f} | "
              f"Δ={(lm_mean-cf_mean):+.4f} | p={p:.4f}")

    return dict(
        arm=arm_name,
        catalog_size=int(train_df['item_id'].nunique()),
        n_users_eval=len(eval_users),
        n_users_train_lm=len(train_users),
        K_values=K_values,
        rows=rows,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_users', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dataset', default='toys')
    ap.add_argument('--catalog_subsample_to', type=int, default=1594,
                    help='Beauty-equivalent catalog size for arm 2')
    ap.add_argument('--quick', action='store_true',
                    help='Smoke test with K=[50,100] only')
    ap.add_argument('--out', default='experiments/logs/cikm_controlled_recall_sweep.json')
    args = ap.parse_args()

    K_values = [50, 100] if args.quick else [20, 50, 100, 150, 200, 300]
    print(f"Recall sweep K values: {K_values}")

    data_path = project_root / 'data' / 'processed' / f'amazon_{args.dataset}_sampled'
    train_df = pd.read_parquet(data_path / 'train.parquet').astype({'user_id': int, 'item_id': int})
    test_df = pd.read_parquet(data_path / 'test.parquet').astype({'user_id': int, 'item_id': int})

    t0 = time.time()
    arm1 = run_arm('full_catalog', train_df, test_df, K_values, args.n_users, seed=args.seed)

    train2, test2 = subsample_catalog(train_df, test_df, args.catalog_subsample_to, seed=args.seed)
    arm2 = run_arm(f'subsampled_catalog_{args.catalog_subsample_to}',
                   train2, test2, K_values, args.n_users, seed=args.seed)

    summary = dict(
        experiment='cikm_controlled_recall_sweep_P0A',
        dataset=args.dataset,
        n_users_target=args.n_users,
        seed=args.seed,
        timestamp=datetime.now().isoformat(timespec='seconds'),
        K_values=K_values,
        arms=[arm1, arm2],
        elapsed_sec=time.time() - t0,
    )

    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    for arm in summary['arms']:
        print(f"\n{arm['arm']} (catalog={arm['catalog_size']}):")
        print(f"  {'K':>4} {'recall%':>9} {'CF NDCG':>10} {'LM NDCG':>10} {'Δ':>10} {'p':>10}")
        for r in arm['rows']:
            print(f"  {r['K']:>4} {r['recall_mean']*100:>8.2f}% "
                  f"{r['cf_ndcg']:>10.4f} {r['lm_ndcg']:>10.4f} "
                  f"{r['delta']:>+10.4f} {r['wilcoxon_p']:>10.4f}")

    print(f"\nSaved: {out_path}")
    print(f"Elapsed: {summary['elapsed_sec']:.1f}s")


if __name__ == '__main__':
    main()
