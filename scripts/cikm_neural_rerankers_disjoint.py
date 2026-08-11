#!/usr/bin/env python3
"""
CIKM 2026 — Neural Rerankers Under Disjoint Train/Eval Split (post-leak fix)
=============================================================================

Replaces the original `kdd_neural_reranker_baselines.py` Table 9 results.

Why
---
The leak diagnostic (`cikm_lambdamart_leak_diagnostic.json`) showed that
the original LambdaMART NDCG of .0464 on Toys collapses to .0050 once the
training-user / evaluation-user overlap is removed. The "8.8% boundary"
in `falsification.tex` §4.4 is exactly the boundary between datasets that
ship with a `val.parquet` (Beauty / Movies / Electronics — clean) and those
that don't (Toys / Sports / Office — fall back to the leaky split). A
methodological artefact, not a structural threshold.

Protocol — Clean Disjoint (Protocol C from diagnostic)
------------------------------------------------------
For every dataset, three disjoint user pools:

  1. CF retrieval is fit on ALL training interactions (unchanged).
  2. RERANKER TRAIN POOL: 250 users drawn from train_df users that are
     NOT in the test set. Labels = each user's LAST training interaction
     (held out during retrieval). This is a clean supervised signal that
     does not touch the evaluation users.
  3. EVAL POOL: the standard 500 sampled test users.

Three rerankers per dataset:
  - LambdaMART (LightGBM, lambdarank objective)
  - RankNet (pairwise neural)
  - MLP (pointwise neural)
Plus CF baseline and Oracle ceiling.

K = 100 (matches the paper's main protocol).

Usage
-----
    python scripts/cikm_neural_rerankers_disjoint.py --n_users 500
    python scripts/cikm_neural_rerankers_disjoint.py --datasets beauty,movies,electronics,toys,sports,office
    python scripts/cikm_neural_rerankers_disjoint.py --quick   # smoke test, 100 users, 2 datasets

Output
------
    experiments/logs/cikm_neural_rerankers_disjoint.json
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

import argparse
import json
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Reuse identical CF + features + LambdaMART from the P0-A script
from cikm_controlled_recall_sweep import (
    CFSVD, build_features, LambdaMART,
    ndcg_at_k, safe_wilcoxon, bootstrap_ci,
)


# ============================================================================
# RankNet — pairwise neural ranker (matches kdd_neural_reranker_baselines.py)
# ============================================================================

class RankNetModel(nn.Module):
    def __init__(self, in_dim, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RankNet:
    def __init__(self, h=64, lr=1e-3, n_epochs=20, device=None, seed=42):
        self.h, self.lr, self.n_epochs, self.seed = h, lr, n_epochs, seed
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def fit(self, X, y, groups):
        torch.manual_seed(self.seed)
        in_dim = X.shape[1]
        self.model = RankNetModel(in_dim, self.h).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        x1s, x2s = [], []
        offset = 0
        for g in groups:
            gf = X[offset:offset + g]; gl = y[offset:offset + g]
            pos_idx = np.where(gl == 1)[0]; neg_idx = np.where(gl == 0)[0]
            for pi in pos_idx:
                for ni in neg_idx[:5]:
                    x1s.append(gf[pi]); x2s.append(gf[ni])
            offset += g

        if not x1s:
            self.model = None
            return

        x1 = torch.tensor(np.array(x1s), dtype=torch.float32, device=self.device)
        x2 = torch.tensor(np.array(x2s), dtype=torch.float32, device=self.device)

        for _ in range(self.n_epochs):
            self.model.train()
            perm = torch.randperm(len(x1))
            for s in range(0, len(x1), 512):
                idx = perm[s:s + 512]
                s1 = self.model(x1[idx]); s2 = self.model(x2[idx])
                loss = -F.logsigmoid(s1 - s2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self.model.eval()

    def predict(self, X):
        if self.model is None:
            return np.zeros(len(X))
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.model(t).cpu().numpy()


# ============================================================================
# MLP — pointwise neural ranker
# ============================================================================

class MLP:
    def __init__(self, h=64, lr=1e-3, n_epochs=20, device=None, seed=42):
        self.h, self.lr, self.n_epochs, self.seed = h, lr, n_epochs, seed
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def fit(self, X, y, groups):
        torch.manual_seed(self.seed)
        in_dim = X.shape[1]
        self.model = nn.Sequential(
            nn.Linear(in_dim, self.h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(self.h, self.h), nn.ReLU(),
            nn.Linear(self.h, 1),
        ).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)
        for _ in range(self.n_epochs):
            self.model.train()
            perm = torch.randperm(len(xt))
            for s in range(0, len(xt), 512):
                idx = perm[s:s + 512]
                pred = self.model(xt[idx]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(pred, yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        self.model.eval()

    def predict(self, X):
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.model(t).squeeze(-1).cpu().numpy()


# ============================================================================
# Build clean training set (Protocol C):
#   train_users disjoint from eval_users
#   labels = each train user's last training interaction (held-out from history)
# ============================================================================

def build_clean_training_data(train_df, user_history, eval_users, K, cf,
                              item_pop, n_train=250, seed=42):
    """Return (X, y, groups, n_train_users, pos_rate)."""
    rng = np.random.default_rng(seed)
    eval_set = set(eval_users)

    # last interaction per user from train_df
    last_train = train_df.sort_values('user_id').groupby('user_id').tail(1)
    held_out = dict(zip(last_train['user_id'].astype(int).values,
                        last_train['item_id'].astype(int).values))

    pool = [u for u, h in user_history.items()
            if u not in eval_set and len(h) >= 4 and u in held_out]
    if len(pool) < 50:
        return None

    train_users = list(rng.choice(pool, size=min(n_train, len(pool)), replace=False))

    Xs, ys, groups = [], [], []
    for uid in tqdm(train_users, desc='    train feats', leave=False):
        ho_item = held_out[uid]
        hist = [h for h in user_history.get(uid, []) if h != ho_item]
        cands, scores = cf.topk(uid, K, hist)
        if len(cands) == 0:
            continue
        feats = build_features(uid, cands, scores, cf, user_history, item_pop)
        labels = np.array([1 if c == ho_item else 0 for c in cands], dtype=np.int32)
        Xs.append(feats); ys.append(labels); groups.append(len(cands))

    if not Xs:
        return None
    X = np.vstack(Xs); y = np.concatenate(ys)
    return X, y, groups, len(train_users), float(y.mean())


# ============================================================================
# Single-dataset run
# ============================================================================

def run_dataset(name, ds_path, args):
    print(f"\n{'='*72}\nDATASET: {name.upper()}\n{'='*72}")

    train_df = pd.read_parquet(ds_path / 'train.parquet').astype({'user_id': int, 'item_id': int})
    test_df = pd.read_parquet(ds_path / 'test.parquet').astype({'user_id': int, 'item_id': int})
    has_val = (ds_path / 'val.parquet').exists()
    print(f"  catalog={train_df['item_id'].nunique()} "
          f"train={len(train_df)} test_users={test_df['user_id'].nunique()} "
          f"has_val={has_val}")

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cf = CFSVD(n_factors=128); cf.fit(train_df)
    item_pop = train_df['item_id'].value_counts().to_dict()
    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()

    eligible = [u for u in test_gt if u in user_history and len(user_history[u]) >= 3]
    n_eval = min(args.n_users, len(eligible))
    eval_users = list(rng.choice(eligible, size=n_eval, replace=False))
    print(f"  eval_users={len(eval_users)}")

    # CF baseline + Oracle on eval set
    cf_ndcgs, oracle_ndcgs, recalls = [], [], []
    user_cands, user_scores = {}, {}
    for uid in eval_users:
        cands, scores = cf.topk(uid, args.K, user_history.get(uid, []))
        user_cands[uid] = cands; user_scores[uid] = scores
        gt = test_gt.get(uid, []); gt_set = set(gt)
        recalls.append(len(set(cands) & gt_set) / max(len(gt_set), 1))
        cf_ndcgs.append(ndcg_at_k(cands, gt, k=10))
        relevant = [c for c in cands if c in gt_set]
        irrelevant = [c for c in cands if c not in gt_set]
        oracle_ndcgs.append(ndcg_at_k(relevant + irrelevant, gt, k=10))

    cf_mean, cf_lo, cf_hi = bootstrap_ci(cf_ndcgs, seed=0)
    or_mean, or_lo, or_hi = bootstrap_ci(oracle_ndcgs, seed=1)
    recall_mean = float(np.mean(recalls))
    print(f"  recall@{args.K}={recall_mean*100:.2f}%  "
          f"CF NDCG={cf_mean:.4f}  Oracle NDCG={or_mean:.4f}")

    # Clean training data (Protocol C)
    print("  Building clean training data (disjoint pool, held-out labels)...")
    train_pack = build_clean_training_data(
        train_df, user_history, eval_users, args.K, cf, item_pop,
        n_train=args.n_train, seed=args.seed)
    if train_pack is None:
        print("  SKIP: insufficient training pool")
        return dict(dataset=name, error='insufficient_training_pool',
                    catalog=int(train_df['item_id'].nunique()),
                    n_eval=len(eval_users), recall=recall_mean,
                    cf_ndcg=cf_mean, oracle_ndcg=or_mean)
    X, y, groups, n_train_users, pos_rate = train_pack
    print(f"    train_users={n_train_users} rows={len(X)} pos_rate={pos_rate:.4f}")

    rerankers = {}

    def eval_reranker(name, model):
        scores = []
        for uid in eval_users:
            cands = user_cands[uid]
            if not cands:
                scores.append(0.0); continue
            feats = build_features(uid, cands, user_scores[uid], cf,
                                   user_history, item_pop)
            preds = model.predict(feats)
            order = np.argsort(preds)[::-1]
            scores.append(ndcg_at_k([cands[i] for i in order],
                                     test_gt.get(uid, []), k=10))
        m, lo, hi = bootstrap_ci(scores, seed=2)
        p = safe_wilcoxon(scores, cf_ndcgs)
        eta = m / recall_mean if recall_mean > 0 else 0.0
        print(f"  {name:>11} NDCG={m:.4f} [{lo:.4f},{hi:.4f}] "
              f"Δ={(m-cf_mean):+.4f} p={p:.4f} η={eta*100:.1f}%")
        return dict(ndcg=m, ci=[lo, hi], delta=m - cf_mean,
                    wilcoxon_p=p, eta=eta, scores_len=len(scores))

    # LambdaMART
    print("  Training LambdaMART...")
    try:
        lm = LambdaMART(); lm.fit(X, y, groups)
        rerankers['LambdaMART'] = eval_reranker('LambdaMART', lm)
    except Exception as e:
        print(f"    LambdaMART error: {e}")
        rerankers['LambdaMART'] = dict(error=str(e))

    # RankNet
    print("  Training RankNet...")
    try:
        rn = RankNet(h=64, lr=1e-3, n_epochs=args.epochs, seed=args.seed)
        rn.fit(X, y, groups)
        if rn.model is None:
            rerankers['RankNet'] = dict(error='no_pairwise_samples')
        else:
            rerankers['RankNet'] = eval_reranker('RankNet', rn)
    except Exception as e:
        print(f"    RankNet error: {e}")
        traceback.print_exc()
        rerankers['RankNet'] = dict(error=str(e))

    # MLP
    print("  Training MLP...")
    try:
        mlp = MLP(h=64, lr=1e-3, n_epochs=args.epochs, seed=args.seed)
        mlp.fit(X, y, groups)
        rerankers['MLP'] = eval_reranker('MLP', mlp)
    except Exception as e:
        print(f"    MLP error: {e}")
        rerankers['MLP'] = dict(error=str(e))

    return dict(
        dataset=name,
        catalog=int(train_df['item_id'].nunique()),
        has_val_parquet=has_val,
        n_eval=len(eval_users),
        n_train_users=n_train_users,
        n_train_rows=int(len(X)),
        train_pos_rate=pos_rate,
        K=args.K,
        recall=recall_mean,
        cf_ndcg=cf_mean, cf_ci=[cf_lo, cf_hi],
        oracle_ndcg=or_mean, oracle_ci=[or_lo, or_hi],
        rerankers=rerankers,
    )


# ============================================================================
# Main
# ============================================================================

DEFAULT_DATASETS = ['beauty', 'movies', 'electronics', 'toys', 'sports', 'office']
DATASET_PATHS = {
    'beauty': 'amazon_beauty_sampled',
    'movies': 'amazon_movies_sampled',
    'electronics': 'amazon_electronics_sampled',
    'toys': 'amazon_toys_sampled',
    'sports': 'amazon_sports_sampled',
    'office': 'amazon_office_sampled',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_users', type=int, default=500)
    ap.add_argument('--n_train', type=int, default=250)
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--datasets', default=','.join(DEFAULT_DATASETS))
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='experiments/logs/cikm_neural_rerankers_disjoint.json')
    args = ap.parse_args()

    if args.quick:
        args.n_users = 100; args.n_train = 100
        args.datasets = 'beauty,toys'

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    print(f"Datasets: {datasets}")
    print(f"n_users={args.n_users} n_train={args.n_train} K={args.K} seed={args.seed}")

    t0 = time.time()
    by_dataset = {}
    for name in datasets:
        path = project_root / 'data' / 'processed' / DATASET_PATHS[name]
        try:
            by_dataset[name] = run_dataset(name, path, args)
        except Exception as e:
            print(f"  FATAL on {name}: {e}")
            traceback.print_exc()
            by_dataset[name] = dict(dataset=name, error=str(e))

    summary = dict(
        experiment='cikm_neural_rerankers_disjoint',
        protocol='Protocol C: train_users disjoint from eval_users; labels = held-out last train interaction',
        timestamp=datetime.now().isoformat(timespec='seconds'),
        config=vars(args),
        results=by_dataset,
        elapsed_sec=time.time() - t0,
    )

    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Final formatted table
    print(f"\n{'='*88}\nFINAL TABLE — Disjoint Protocol\n{'='*88}")
    print(f"{'Dataset':<14}{'Catalog':>9}{'Recall':>9}{'CF':>10}"
          f"{'LambdaMART':>14}{'RankNet':>14}{'MLP':>14}")
    for name in datasets:
        r = by_dataset.get(name, {})
        if 'error' in r and 'rerankers' not in r:
            print(f"{name:<14}  ERROR: {r.get('error')}")
            continue
        rer = r.get('rerankers', {})
        def cell(k):
            v = rer.get(k, {})
            if 'error' in v:
                return 'err'
            return f"{v['ndcg']:.4f}/{v['wilcoxon_p']:.3f}"
        print(f"{name:<14}{r.get('catalog',0):>9}{r.get('recall',0)*100:>8.2f}%"
              f"{r.get('cf_ndcg',0):>10.4f}"
              f"{cell('LambdaMART'):>14}{cell('RankNet'):>14}{cell('MLP'):>14}")

    print(f"\nSaved: {out_path}")
    print(f"Elapsed: {summary['elapsed_sec']:.1f}s")


if __name__ == '__main__':
    main()
