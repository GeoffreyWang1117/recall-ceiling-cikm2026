#!/usr/bin/env python3
"""
CIKM v4 — Multi-positive (last-5) evaluation on Beauty/Movies/Electronics.

Reviewer concern (R-Empiricist v5): LOO with |Y_u|=1 makes the recall
ceiling Recall@K-tight; a multi-positive evaluation tests whether the
gap survives.

Protocol: each user's LAST 5 chronological interactions are positives;
the rest of the interactions are training history. We compute:
  - Recall@100 with CF-SVD (the retrieval baseline used throughout)
  - Hit@10, NDCG@10, MAP@10 under realistic (CF-SVD retrieval) and oracle
    (positives injected) conditions
  - Bootstrap 95% CIs, paired Wilcoxon CF vs oracle gap

Output: experiments/logs/cikm_multipositive_last5_v4.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from cikm_controlled_recall_sweep import CFSVD, bootstrap_ci, safe_wilcoxon

DATA = REPO / "data" / "processed"
LOG = REPO / "experiments" / "logs"


def ndcg_at_k_multi(ranked: List[int], gt_set: set[int], k: int) -> float:
    """NDCG@k for multi-positive ground-truth."""
    dcg = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in gt_set:
            dcg += 1.0 / np.log2(i + 1)
    # ideal DCG: min(|gt_set|, k) ones at the top
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gt_set), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def map_at_k_multi(ranked: List[int], gt_set: set[int], k: int) -> float:
    if not gt_set:
        return 0.0
    hits, precs = 0, 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in gt_set:
            hits += 1
            precs += hits / i
    return precs / min(len(gt_set), k)


def hit_at_k(ranked: List[int], gt_set: set[int], k: int) -> int:
    return int(any(it in gt_set for it in ranked[:k]))


def recall_at_k(ranked: List[int], gt_set: set[int], k: int) -> float:
    if not gt_set:
        return 0.0
    return sum(1 for it in ranked[:k] if it in gt_set) / len(gt_set)


def build_last_n_split(ds: str, n_positives: int = 5):
    """
    Reconstruct per-user history with last-N held out as multi-positive eval set.
    Concatenate train+val+test, sort by timestamp if available else parquet row order.
    """
    parts = []
    for split in ["train", "val", "test"]:
        p = DATA / ds / f"{split}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    # if timestamp column exists, sort
    ts_col = None
    for c in ["timestamp", "ts", "time"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is not None:
        df = df.sort_values(["user_id", ts_col])
    # for each user, last-n as eval positives; rest as training
    df_train_rows, df_eval = [], {}
    for uid, group in df.groupby("user_id", sort=False):
        items = group["item_id"].tolist()
        if len(items) <= n_positives + 2:  # need at least 2 training items
            continue
        train_items, eval_items = items[:-n_positives], items[-n_positives:]
        df_train_rows.extend([(uid, it) for it in train_items])
        df_eval[uid] = set(eval_items)
    train_df = pd.DataFrame(df_train_rows, columns=["user_id", "item_id"])
    return train_df, df_eval


def evaluate(ds: str, n_users: int, K: int, n_positives: int = 5, seed: int = 42) -> Dict:
    print(f"\n=== {ds} last-{n_positives} ===")
    train_df, eval_gt = build_last_n_split(ds, n_positives)
    print(f"  train_rows={len(train_df)}, eval_users={len(eval_gt)}")
    rng = np.random.RandomState(seed)
    users = list(eval_gt.keys())
    if len(users) > n_users:
        users = list(rng.choice(users, n_users, replace=False))

    print("  fitting CFSVD ...")
    cf = CFSVD(n_factors=128)
    cf.fit(train_df)

    metrics_real = {"recall100": [], "hit10": [], "ndcg10": [], "map10": []}
    metrics_oracle = {"hit10": [], "ndcg10": [], "map10": []}

    for uid in tqdm(users, desc=f"{ds} eval"):
        gt = eval_gt[uid]
        train_items = set(train_df[train_df["user_id"] == uid]["item_id"].tolist())
        # Realistic: CF retrieves K candidates (no GT injection); ranking = CF score order
        cands_real, scores_real = cf.topk(int(uid), K, train_items)
        metrics_real["recall100"].append(recall_at_k(cands_real, gt, K))
        metrics_real["hit10"].append(hit_at_k(cands_real, gt, 10))
        metrics_real["ndcg10"].append(ndcg_at_k_multi(cands_real, gt, 10))
        metrics_real["map10"].append(map_at_k_multi(cands_real, gt, 10))
        # Oracle PROTOCOL (matches LLMRank/Beyond Utility setup): inject all GT
        # items into the candidate set, then rerank K candidates by CF score
        # (drop CF candidates to make room as needed). We use CF as the reranker
        # here as a *baseline* oracle; LLM rerankers would replace CF on this set.
        # The point is: the candidate set CONTAINS the positives, so reranker
        # can place them at top if it knows preference signal.
        missing = list(gt - set(cands_real))
        cands_oracle = list(cands_real)
        # replace the K-tail with missing GT items
        for m in missing:
            if len(cands_oracle) < K:
                cands_oracle.append(m)
            else:
                cands_oracle[-1] = m
                cands_oracle.append(None)  # placeholder; will be dropped
                cands_oracle = [c for c in cands_oracle if c is not None][:K]
        # rank cands_oracle by CF score (using the same CF model's scoring)
        try:
            uidx = cf.user_to_idx.get(int(uid))
            if uidx is not None:
                u_emb = cf.user_factors[uidx]
                cf_scores = {}
                for c in cands_oracle:
                    if int(c) in cf.item_to_idx:
                        cf_scores[c] = float(u_emb @ cf.item_factors[cf.item_to_idx[int(c)]])
                    else:
                        cf_scores[c] = -1e9
                # GT items get a +small bonus to break ties consistently with what
                # an oracle-protocol "the reranker sees the positives" implies;
                # without this, CF-score on injected positives may be < CF-score
                # on top-K candidates, leaving NDCG unchanged. The oracle
                # protocol in published LLM-rerank papers assumes the reranker
                # CAN place injected positives at top; we report two variants:
                cands_oracle_cf = sorted(cands_oracle, key=lambda c: -cf_scores[c])
            else:
                cands_oracle_cf = cands_oracle
        except Exception:
            cands_oracle_cf = cands_oracle
        # additionally: ideal-oracle = place GT at top (Theorem 1 ceiling)
        cands_oracle_ideal = list(gt) + [c for c in cands_oracle if c not in gt]
        cands_oracle_ideal = cands_oracle_ideal[:K]
        metrics_oracle["hit10"].append(hit_at_k(cands_oracle_cf, gt, 10))
        metrics_oracle["ndcg10"].append(ndcg_at_k_multi(cands_oracle_cf, gt, 10))
        metrics_oracle["map10"].append(map_at_k_multi(cands_oracle_cf, gt, 10))
        # also track the Theorem 1 ceiling (ideal placement)
        metrics_oracle.setdefault("ndcg10_ceiling", []).append(
            ndcg_at_k_multi(cands_oracle_ideal, gt, 10)
        )

    out = {"dataset": ds, "n_users": len(users), "K": K, "n_positives": n_positives}
    out["realistic"] = {}
    out["oracle"] = {}
    for m in metrics_real:
        a = np.array(metrics_real[m])
        mean, lo, hi = bootstrap_ci(a, n=2000)
        out["realistic"][m] = dict(mean=float(mean), ci_lo=float(lo), ci_hi=float(hi))
    for m in metrics_oracle:
        a = np.array(metrics_oracle[m])
        mean, lo, hi = bootstrap_ci(a, n=2000)
        out["oracle"][m] = dict(mean=float(mean), ci_lo=float(lo), ci_hi=float(hi))
    # Theorem 1 ceiling: NDCG@10 of ideal placement (gt at top of injected set)
    if "ndcg10_ceiling" in metrics_oracle:
        a = np.array(metrics_oracle["ndcg10_ceiling"])
        mean, lo, hi = bootstrap_ci(a, n=2000)
        out["ceiling_ndcg10"] = dict(mean=float(mean), ci_lo=float(lo), ci_hi=float(hi))
    # gap and paired-Wilcoxon (oracle vs realistic, per user)
    for m in ("hit10", "ndcg10", "map10"):
        a_r = np.array(metrics_real[m]); a_o = np.array(metrics_oracle[m])
        try:
            _, p = safe_wilcoxon(a_o, a_r)
        except Exception:
            p = None
        gap = 1.0 - (a_r.mean() / a_o.mean() if a_o.mean() > 0 else 0.0)
        out[f"gap_{m}"] = float(gap)
        out[f"p_paired_{m}"] = float(p) if p is not None else None

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["amazon_beauty_sampled","amazon_movies_sampled","amazon_electronics_sampled"])
    ap.add_argument("--n_users", type=int, default=500)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--n_positives", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(LOG / "cikm_multipositive_last5_v4.json"))
    args = ap.parse_args()
    out = {}
    for ds in args.datasets:
        try:
            out[ds] = evaluate(ds, args.n_users, args.K, args.n_positives, args.seed)
        except Exception as e:
            import traceback; traceback.print_exc()
            out[ds] = {"error": str(e)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n>>> wrote {args.out}")
    print("\n=== SUMMARY: multi-positive last-5 eval ===")
    print(f"{'Dataset':22s} {'Real R@100':>10s} {'Real NDCG':>10s} {'OracleCF':>10s} {'Ceiling':>10s} {'Gap%':>6s}")
    for ds, r in out.items():
        if "realistic" not in r: continue
        ceil = r.get("ceiling_ndcg10",{}).get("mean", float("nan"))
        gap_ceiling = 1.0 - r["realistic"]["ndcg10"]["mean"]/ceil if ceil > 0 else 0
        print(f"{ds:22s} {100*r['realistic']['recall100']['mean']:>9.2f}% "
              f"{r['realistic']['ndcg10']['mean']:>10.4f} "
              f"{r['oracle']['ndcg10']['mean']:>10.4f} "
              f"{ceil:>10.4f} "
              f"{100*gap_ceiling:>5.1f}%")


if __name__ == "__main__":
    main()
