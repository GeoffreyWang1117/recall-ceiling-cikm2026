#!/usr/bin/env python3
"""
CIKM v6 — Closed-catalog robustness check.

Reviewer concern: the low CF-SVD Recall@100 on Amazon could partially be a
"the held-out test item is not in the retrieval catalog" artifact (open-catalog
setting), not a true reranking-stage failure.

This script filters the eval users so the held-out test item is also visible
to the retriever, i.e. it appears in the training-split catalog. We then
recompute Recall@100, CF NDCG@10, and the injection-oracle NDCG@10 on this
closed-catalog-only subset to verify the ceiling persists.

Output: experiments/logs/cikm_closed_catalog_v6.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from cikm_controlled_recall_sweep import CFSVD, ndcg_at_k, bootstrap_ci

DATA = REPO / "data" / "processed"
LOG = REPO / "experiments" / "logs"


def evaluate(ds: str, n_users: int, K: int, seed: int = 42):
    print(f"\n=== {ds} ===")
    train = pd.read_parquet(DATA / ds / "train.parquet")
    val_path = DATA / ds / "val.parquet"
    val = pd.read_parquet(val_path) if val_path.exists() else None
    test = pd.read_parquet(DATA / ds / "test.parquet")

    train_items = set(train["item_id"].unique())
    print(f"  train_catalog = {len(train_items)} items")
    print(f"  test users    = {test['user_id'].nunique()}")

    test_gt = test.set_index("user_id")["item_id"].to_dict()
    user_hist = train.groupby("user_id")["item_id"].apply(list).to_dict()
    if val is not None:
        val_hist = val.groupby("user_id")["item_id"].apply(list).to_dict()
    else:
        val_hist = {}

    rng = np.random.RandomState(seed)
    all_users = sorted(test_gt.keys())
    chosen = sorted(rng.choice(all_users, min(n_users, len(all_users)), replace=False))

    # split into open-catalog vs closed-catalog
    closed = [u for u in chosen if test_gt[u] in train_items]
    open_users = [u for u in chosen if test_gt[u] not in train_items]
    print(f"  sampled        = {len(chosen)} users (seed={seed})")
    print(f"  closed-catalog = {len(closed)} users (held-out item IN train catalog)")
    print(f"  open-catalog   = {len(open_users)} users (held-out item NOT in train)")

    print("  fitting CFSVD ...")
    cf = CFSVD(n_factors=128)
    cf.fit(train)

    def eval_subset(users, label):
        if not users:
            return None
        recall_arr, cf_ndcg_arr, oracle_ndcg_arr = [], [], []
        for uid in users:
            gt = test_gt[uid]
            exclude = set(user_hist.get(uid, [])) | set(val_hist.get(uid, []))
            cands, _ = cf.topk(int(uid), K, exclude)
            recall_arr.append(1.0 if gt in cands else 0.0)
            cf_ndcg_arr.append(ndcg_at_k(cands, gt, k=10))
            # injection-oracle: inject gt at top
            inj = [gt] + [c for c in cands if c != gt][:K-1]
            oracle_ndcg_arr.append(ndcg_at_k(inj, gt, k=10))
        ra = np.array(recall_arr); ca = np.array(cf_ndcg_arr); oa = np.array(oracle_ndcg_arr)
        r_m, r_lo, r_hi = bootstrap_ci(ra, n=2000)
        c_m, c_lo, c_hi = bootstrap_ci(ca, n=2000)
        o_m, o_lo, o_hi = bootstrap_ci(oa, n=2000)
        return {
            "n_users": len(users),
            "recall_at_100": {"mean": float(r_m), "ci": [float(r_lo), float(r_hi)]},
            "cf_ndcg10":     {"mean": float(c_m), "ci": [float(c_lo), float(c_hi)]},
            "injection_oracle_ndcg10": {"mean": float(o_m), "ci": [float(o_lo), float(o_hi)]},
            "zero_recall_rate": float(1.0 - r_m),
        }

    return {
        "dataset": ds,
        "train_catalog_size": len(train_items),
        "total_users_sampled": len(chosen),
        "open_catalog_pct": 100*len(open_users)/max(len(chosen), 1),
        "full_sample": eval_subset(chosen, "full"),
        "closed_catalog_only": eval_subset(closed, "closed"),
        "open_catalog_only": eval_subset(open_users, "open") if open_users else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["amazon_beauty_sampled","amazon_movies_sampled","amazon_electronics_sampled"])
    ap.add_argument("--n_users", type=int, default=500)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(LOG / "cikm_closed_catalog_v6.json"))
    args = ap.parse_args()
    out = {}
    for ds in args.datasets:
        try:
            out[ds] = evaluate(ds, args.n_users, args.K, args.seed)
        except Exception as e:
            import traceback; traceback.print_exc()
            out[ds] = {"error": str(e)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n>>> wrote {args.out}")
    print("\n=== SUMMARY: closed-catalog robustness ===")
    print(f"{'Dataset':22s} {'open%':>6s} | {'FULL R@100':>10s} {'CLOSED R@100':>14s} | {'FULL CF':>9s} {'CLOSED CF':>10s} | {'FULL O':>9s} {'CLOSED O':>10s}")
    for ds, r in out.items():
        if "full_sample" not in r: continue
        f, c = r["full_sample"], r["closed_catalog_only"]
        print(f"{ds:22s} {r['open_catalog_pct']:>5.1f}% | "
              f"{100*f['recall_at_100']['mean']:>9.2f}% {100*c['recall_at_100']['mean']:>13.2f}% | "
              f"{f['cf_ndcg10']['mean']:>9.4f} {c['cf_ndcg10']['mean']:>10.4f} | "
              f"{f['injection_oracle_ndcg10']['mean']:>9.4f} {c['injection_oracle_ndcg10']['mean']:>10.4f}")


if __name__ == "__main__":
    main()
