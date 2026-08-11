#!/usr/bin/env python3
"""
CIKM 2026 — Diagnostic: Does the original LambdaMART protocol leak?

Goal
----
Decide whether the .0464 vs .0050 LambdaMART discrepancy between
`kdd_neural_reranker_baselines.json` (Toys, K=100, claimed p<.0001)
and `cikm_controlled_recall_sweep.json` (Toys, K=150 ≈ same recall, p=.834)
is explained by train/eval user overlap in the original protocol.

Method
------
On a single dataset (Toys), train+evaluate LambdaMART under three protocols
that differ ONLY in the user partition. All other code (features, K, model
params, RNG seed) is identical.

  Protocol A — ORIGINAL (overlap):
    train_users = sampled_users[:200]
    eval_users  = sampled_users      # same 500, train ⊂ eval
    Replicates kdd_neural_reranker_baselines.py logic for datasets w/o val.parquet.

  Protocol B — DISJOINT:
    train_users = sampled_users[:250]
    eval_users  = sampled_users[250:]   # disjoint halves
    What cikm_controlled_recall_sweep.py uses.

  Protocol C — TRAIN-ONLY-FROM-NON-TEST (clean):
    train_users drawn from train_df users NOT in the test set
    eval_users  = sampled_users (any 500 with test_gt)
    The properly clean baseline. Distinct user populations.

For each protocol we report LambdaMART NDCG@10, CF NDCG@10, Wilcoxon p,
and (most importantly for diagnosis) LambdaMART NDCG@10 evaluated only on
the train_users — if this is much higher than on held-out users, the
model has overfit/memorised them.

Usage
-----
    python scripts/cikm_diagnose_lambdamart_leak.py --n_users 500 --K 100

Output
------
    experiments/logs/cikm_lambdamart_leak_diagnostic.json
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
from scipy.stats import wilcoxon
from tqdm import tqdm

# Reuse exactly the same CF + features + LambdaMART class definitions used in P0-A,
# so the only thing that changes between protocols is the user partition.
from cikm_controlled_recall_sweep import (
    CFSVD, build_features, LambdaMART,
    ndcg_at_k, safe_wilcoxon, bootstrap_ci,
)


def evaluate(cf, lm, eval_users, K, user_history, item_pop, test_gt):
    """Score all eval users, return per-user (cf_ndcg, lm_ndcg, recall)."""
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
        lm_ndcgs.append(ndcg_at_k([cands[i] for i in order], gt, k=10))
    return cf_ndcgs, lm_ndcgs, recalls


def fit_lambdamart(train_users, K, cf, user_history, item_pop, test_gt):
    Xs, ys, groups = [], [], []
    for uid in tqdm(train_users, desc='    train feats', leave=False):
        cands, scores = cf.topk(uid, K, user_history.get(uid, []))
        if len(cands) == 0:
            continue
        feats = build_features(uid, cands, scores, cf, user_history, item_pop)
        gt = set(test_gt.get(uid, []))
        labels = np.array([1 if c in gt else 0 for c in cands], dtype=np.int32)
        Xs.append(feats); ys.append(labels); groups.append(len(cands))
    X = np.vstack(Xs); y = np.concatenate(ys)
    lm = LambdaMART()
    lm.fit(X, y, groups)
    return lm, len(X), float(y.mean())


def summarize(label, cf_ndcgs, lm_ndcgs, recalls):
    cf_mean, cf_lo, cf_hi = bootstrap_ci(cf_ndcgs, seed=0)
    lm_mean, lm_lo, lm_hi = bootstrap_ci(lm_ndcgs, seed=1)
    p = safe_wilcoxon(lm_ndcgs, cf_ndcgs)
    return dict(
        label=label,
        n=len(cf_ndcgs),
        recall_mean=float(np.mean(recalls)) if recalls else 0.0,
        cf_ndcg=cf_mean, cf_ci=[cf_lo, cf_hi],
        lm_ndcg=lm_mean, lm_ci=[lm_lo, lm_hi],
        delta=lm_mean - cf_mean,
        wilcoxon_p=p,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_users', type=int, default=500)
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dataset', default='toys')
    ap.add_argument('--out', default='experiments/logs/cikm_lambdamart_leak_diagnostic.json')
    args = ap.parse_args()

    print(f"\n{'='*72}\nLambdaMART leak diagnostic\n{'='*72}")
    print(f"  dataset={args.dataset} K={args.K} n_users={args.n_users} seed={args.seed}")

    data_path = project_root / 'data' / 'processed' / f'amazon_{args.dataset}_sampled'
    train_df = pd.read_parquet(data_path / 'train.parquet').astype({'user_id': int, 'item_id': int})
    test_df = pd.read_parquet(data_path / 'test.parquet').astype({'user_id': int, 'item_id': int})

    # Build CF on the full train_df once (same retrieval everywhere)
    cf = CFSVD(n_factors=128); cf.fit(train_df)
    item_pop = train_df['item_id'].value_counts().to_dict()
    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()

    rng = np.random.default_rng(args.seed)

    # Eligible test users (have history >=3 and a test item)
    test_users_all = [u for u in test_gt
                      if u in user_history and len(user_history[u]) >= 3]
    sampled_users = list(rng.permutation(test_users_all)[:args.n_users])
    print(f"  Sampled users: {len(sampled_users)}")

    results = {}
    t0 = time.time()

    # ------------------------------------------------------------------
    # Protocol A — ORIGINAL (overlap)
    # ------------------------------------------------------------------
    print("\n[A] ORIGINAL protocol (train ⊂ eval)")
    train_A = sampled_users[:200]
    eval_A  = sampled_users  # all 500
    lm_A, n_rows_A, pos_A = fit_lambdamart(train_A, args.K, cf, user_history, item_pop, test_gt)
    cfA, lmA, rA = evaluate(cf, lm_A, eval_A, args.K, user_history, item_pop, test_gt)
    sumA = summarize('A_original_overlap', cfA, lmA, rA)
    sumA.update(n_train_users=len(train_A), n_train_rows=n_rows_A, train_pos_rate=pos_A)
    print(f"   eval={sumA['n']} recall={sumA['recall_mean']*100:.2f}% "
          f"CF={sumA['cf_ndcg']:.4f} LM={sumA['lm_ndcg']:.4f} "
          f"Δ={sumA['delta']:+.4f} p={sumA['wilcoxon_p']}")

    # Sub-evaluation: how does protocol A LambdaMART do specifically on the
    # 200 users it was trained on, vs the 300 unseen users?
    train_A_set = set(train_A)
    seen = [(c, l, r) for u, c, l, r in zip(eval_A, cfA, lmA, rA) if u in train_A_set]
    unseen = [(c, l, r) for u, c, l, r in zip(eval_A, cfA, lmA, rA) if u not in train_A_set]
    if seen:
        seen_cf, seen_lm, seen_r = zip(*seen)
        sumA_seen = summarize('A_overlap_seen_users', list(seen_cf), list(seen_lm), list(seen_r))
        print(f"   - on 200 seen users:   LM={sumA_seen['lm_ndcg']:.4f} (CF={sumA_seen['cf_ndcg']:.4f})")
    else:
        sumA_seen = None
    if unseen:
        un_cf, un_lm, un_r = zip(*unseen)
        sumA_unseen = summarize('A_overlap_unseen_users', list(un_cf), list(un_lm), list(un_r))
        print(f"   - on 300 unseen users: LM={sumA_unseen['lm_ndcg']:.4f} (CF={sumA_unseen['cf_ndcg']:.4f})")
    else:
        sumA_unseen = None

    results['protocol_A_original_overlap'] = sumA
    results['protocol_A_seen_users_only'] = sumA_seen
    results['protocol_A_unseen_users_only'] = sumA_unseen

    # ------------------------------------------------------------------
    # Protocol B — DISJOINT halves
    # ------------------------------------------------------------------
    print("\n[B] DISJOINT halves (train ⊥ eval)")
    half = len(sampled_users) // 2
    train_B = sampled_users[:half]
    eval_B  = sampled_users[half:]
    lm_B, n_rows_B, pos_B = fit_lambdamart(train_B, args.K, cf, user_history, item_pop, test_gt)
    cfB, lmB, rB = evaluate(cf, lm_B, eval_B, args.K, user_history, item_pop, test_gt)
    sumB = summarize('B_disjoint', cfB, lmB, rB)
    sumB.update(n_train_users=len(train_B), n_train_rows=n_rows_B, train_pos_rate=pos_B)
    print(f"   eval={sumB['n']} recall={sumB['recall_mean']*100:.2f}% "
          f"CF={sumB['cf_ndcg']:.4f} LM={sumB['lm_ndcg']:.4f} "
          f"Δ={sumB['delta']:+.4f} p={sumB['wilcoxon_p']}")
    results['protocol_B_disjoint'] = sumB

    # ------------------------------------------------------------------
    # Protocol C — train_users from a DIFFERENT user pool (not in test)
    # Toys has 1594 test users out of 1594 train users (LOO), so every
    # train user has a held-out test item. To get a non-test pool, we
    # sample users that we never put in eval_users and use the LAST
    # train interaction (rather than test_gt) as the supervision label.
    # ------------------------------------------------------------------
    print("\n[C] TRAIN-FROM-HELD-OUT (clean: separate user pool, train labels = last train interaction)")
    # Build a held-out label map from the LAST training interaction of each user.
    # Sort within each user by index order (already chronological in source).
    last_train = train_df.sort_values('user_id').groupby('user_id').tail(1)
    held_out_label = dict(zip(last_train['user_id'].values, last_train['item_id'].values))
    # Train pool = users NOT in eval_users (sampled_users), with at least 4 interactions
    eval_set = set(sampled_users)
    candidates_for_train = [u for u, h in user_history.items()
                            if u not in eval_set and len(h) >= 4]
    if len(candidates_for_train) < 50:
        print("   skip: insufficient train pool")
        results['protocol_C_clean'] = None
    else:
        train_C = list(rng.choice(candidates_for_train,
                                   size=min(250, len(candidates_for_train)),
                                   replace=False))
        # Use held-out (last train interaction) as label, and exclude that
        # interaction from CF history during candidate retrieval to avoid leakage.
        # We do this by passing a held-out label dict instead of test_gt.
        Xs, ys, groups = [], [], []
        for uid in tqdm(train_C, desc='    train feats', leave=False):
            ho_item = held_out_label.get(uid)
            if ho_item is None:
                continue
            # Exclude all interacted items + the held-out item itself? No — we WANT
            # the held-out item to be retrievable, since it's our positive label.
            # Exclude interacted items EXCEPT the held-out one.
            history = [h for h in user_history.get(uid, []) if h != ho_item]
            cands, scores = cf.topk(uid, args.K, history)
            if len(cands) == 0:
                continue
            feats = build_features(uid, cands, scores, cf, user_history, item_pop)
            labels = np.array([1 if c == ho_item else 0 for c in cands], dtype=np.int32)
            Xs.append(feats); ys.append(labels); groups.append(len(cands))

        X = np.vstack(Xs); y = np.concatenate(ys)
        lm_C = LambdaMART(); lm_C.fit(X, y, groups)
        cfC, lmC, rC = evaluate(cf, lm_C, sampled_users, args.K, user_history, item_pop, test_gt)
        sumC = summarize('C_clean_separate_pool', cfC, lmC, rC)
        sumC.update(n_train_users=len(train_C), n_train_rows=int(len(X)), train_pos_rate=float(y.mean()))
        print(f"   eval={sumC['n']} recall={sumC['recall_mean']*100:.2f}% "
              f"CF={sumC['cf_ndcg']:.4f} LM={sumC['lm_ndcg']:.4f} "
              f"Δ={sumC['delta']:+.4f} p={sumC['wilcoxon_p']}")
        results['protocol_C_clean'] = sumC

    # ------------------------------------------------------------------
    summary = dict(
        experiment='cikm_lambdamart_leak_diagnostic',
        dataset=args.dataset,
        K=args.K,
        n_users=args.n_users,
        seed=args.seed,
        timestamp=datetime.now().isoformat(timespec='seconds'),
        protocols=results,
        elapsed_sec=time.time() - t0,
    )
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*72}\nDIAGNOSIS SUMMARY\n{'='*72}")
    print(f"  Protocol A (overlap, all 500):           LM={sumA['lm_ndcg']:.4f}  p={sumA['wilcoxon_p']}")
    if sumA_seen:    print(f"  Protocol A — 200 SEEN train users:       LM={sumA_seen['lm_ndcg']:.4f}")
    if sumA_unseen:  print(f"  Protocol A — 300 UNSEEN users (held):    LM={sumA_unseen['lm_ndcg']:.4f}")
    print(f"  Protocol B (disjoint halves, n={sumB['n']}):    LM={sumB['lm_ndcg']:.4f}  p={sumB['wilcoxon_p']}")
    if results['protocol_C_clean']:
        sC = results['protocol_C_clean']
        print(f"  Protocol C (clean separate pool, n={sC['n']}): LM={sC['lm_ndcg']:.4f}  p={sC['wilcoxon_p']}")

    paper_value = 0.0464  # Toys LambdaMART NDCG from kdd_neural_reranker_baselines.json
    print(f"\n  Paper-reported LambdaMART NDCG (Toys K=100): {paper_value:.4f}")
    print(f"  Saved: {out_path}")
    print(f"  Elapsed: {summary['elapsed_sec']:.1f}s")


if __name__ == '__main__':
    main()
