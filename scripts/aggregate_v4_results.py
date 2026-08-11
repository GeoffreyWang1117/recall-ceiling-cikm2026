#!/usr/bin/env python3
"""
Post-hoc aggregation of n500 multi-API rebuttal experiment.
Computes ndcg@10 and recall@10 for every (dataset, model) checkpoint, plus
CF baseline. Writes a JSON summary and prints a comparison table.

Reproduces the 500-user sample using SEED=42 + np.random.choice (matching the
sampling block in scripts/rebuttal_multiapi_n500.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

PROJECT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT / "experiments/logs/checkpoints"
SEED, K, TOP_K, N_FACTORS = 42, 100, 10, 128

DATASETS = {
    "movies": "data/processed/amazon_movies_sampled",
    "beauty": "data/processed/amazon_beauty_sampled",
    "electronics": "data/processed/amazon_electronics_sampled",
}

VARIANTS = [
    "deepseek-v4-pro-nothink", "deepseek-v4-pro-think",
    "deepseek-v4-flash-nothink", "deepseek-v4-flash-think",
    "deepseek-chat", "gpt-4o-mini",
    "kimi-k2-turbo-preview", "qwen_qwen3-32b",
    "llama-3.3-70b-versatile",
]

PRICES = {  # per 1M tokens (USD): (in, out)
    "deepseek-v4-pro-nothink": (0.435, 0.87),
    "deepseek-v4-pro-think":   (0.435, 0.87),
    "deepseek-v4-flash-nothink": (0.14, 0.28),
    "deepseek-v4-flash-think":   (0.14, 0.28),
    "deepseek-chat": (0.27, 1.10),
    "gpt-4o-mini": (0.15, 0.60),
    "kimi-k2-turbo-preview": (0.60, 2.50),
    "qwen_qwen3-32b": (0.10, 0.30),
    "llama-3.3-70b-versatile": (0.05, 0.06),
}


def ndcg_at_k(ranked_ids, gt_ids, k=10):
    dcg  = sum(1 / np.log2(i + 2) for i, x in enumerate(ranked_ids[:k]) if x in gt_ids)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(gt_ids))))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, gt_ids, k=10):
    return len(set(ranked_ids[:k]) & set(gt_ids)) / len(gt_ids) if gt_ids else 0.0


def bootstrap_ci(arr, n=2000, alpha=0.05):
    a = np.array(arr)
    samples = np.random.choice(a, (n, len(a)), replace=True).mean(axis=1)
    return float(a.mean()), float(np.percentile(samples, 100 * alpha / 2)), \
           float(np.percentile(samples, 100 * (1 - alpha / 2)))


class CFRetrieval:
    def fit(self, train_df):
        items = train_df["item_id"].unique()
        users = train_df["user_id"].unique()
        self.item_to_idx = {v: i for i, v in enumerate(items)}
        self.idx_to_item = {i: v for v, i in self.item_to_idx.items()}
        self.user_to_idx = {v: i for i, v in enumerate(users)}
        rows = [self.user_to_idx[u] for u in train_df["user_id"]]
        cols = [self.item_to_idx[i] for i in train_df["item_id"]]
        mat  = csr_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(len(users), len(items)))
        n_comp = min(N_FACTORS, min(mat.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        self.U = svd.fit_transform(mat)
        self.V = svd.components_.T
        self.history = train_df.groupby("user_id")["item_id"].apply(list).to_dict()

    def get_candidates(self, uid, k=100):
        if uid not in self.user_to_idx:
            return [], []
        uf = self.U[self.user_to_idx[uid]]
        scores = self.V @ uf
        for item in self.history.get(uid, []):
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[i] for i in top_idx]


def load_jsonl(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["user_id"])] = r
    return out


def aggregate_dataset(ds_name, data_path):
    print(f"\n{'='*70}\n  Dataset: {ds_name}\n{'='*70}")
    data_dir = PROJECT / data_path
    train = pd.read_parquet(data_dir / "train.parquet")
    test_file = data_dir / "test_loo.parquet" if (data_dir / "test_loo.parquet").exists() else data_dir / "test.parquet"
    test  = pd.read_parquet(test_file)

    cf = CFRetrieval(); cf.fit(train)

    test_map = test.groupby("user_id")["item_id"].apply(list).to_dict()

    # Use the user list actually present in checkpoint files (the experiment's
    # sample at run time), not a re-derived one. Take the union of user_ids
    # from all checkpoints for this dataset, intersected with users that have
    # CF candidates and ground-truth.
    seen_uids: set[str] = set()
    for variant in VARIANTS:
        ckpt = CKPT_DIR / f"multiapi_{variant}_{ds_name}.jsonl"
        if ckpt.exists():
            for line in ckpt.read_text().splitlines():
                if line.strip():
                    seen_uids.add(str(json.loads(line)["user_id"]))
    type_sample = next(iter(test_map)) if test_map else None
    cast = (lambda s: int(s)) if isinstance(type_sample, (int, np.integer)) else (lambda s: s)
    users = []
    for s in seen_uids:
        try:
            u = cast(s)
        except Exception:
            continue
        if u in cf.user_to_idx and test_map.get(u):
            users.append(u)
    print(f"  Users: {len(users)} (from checkpoints), Items: {train['item_id'].nunique()}")

    user_cands = {}
    cf_ndcg10, cf_recall10, cf_recall100 = [], [], []
    for u in users:
        gt = set(test_map[u])
        cands = cf.get_candidates(u, K)
        user_cands[str(u)] = cands
        cf_ndcg10.append(ndcg_at_k(cands, gt, TOP_K))
        cf_recall10.append(recall_at_k(cands, gt, TOP_K))
        cf_recall100.append(recall_at_k(cands, gt, K))

    cf_n_m, cf_n_lo, cf_n_hi = bootstrap_ci(cf_ndcg10)
    cf_r_m, cf_r_lo, cf_r_hi = bootstrap_ci(cf_recall10)
    cf_r100, _, _ = bootstrap_ci(cf_recall100)

    out = {
        "n_users": len(users),
        "CF": {
            "ndcg10_mean": cf_n_m, "ndcg10_ci": [cf_n_lo, cf_n_hi],
            "recall10_mean": cf_r_m, "recall10_ci": [cf_r_lo, cf_r_hi],
            "recall100_mean": cf_r100,
        },
        "models": {},
    }
    print(f"  CF      NDCG@10={cf_n_m:.4f} [{cf_n_lo:.4f},{cf_n_hi:.4f}]  "
          f"Recall@10={cf_r_m:.4f}  Recall@100(ceiling)={cf_r100:.4f}")

    for variant in VARIANTS:
        ckpt = CKPT_DIR / f"multiapi_{variant}_{ds_name}.jsonl"
        recs = load_jsonl(ckpt)
        if not recs:
            continue
        ndcgs, recalls = [], []
        in_tok = out_tok = 0
        for u in users:
            r = recs.get(str(u))
            if r is None:
                continue
            cands = user_cands[str(u)]
            gt = set(test_map[u])
            order = r.get("llm_order", [])
            n_shown = min(30, len(cands))
            order = [i for i in order if 0 <= i < n_shown]
            seen = set(order)
            for i in range(n_shown):
                if i not in seen:
                    order.append(i)
            ranked = [cands[i] for i in order] + cands[n_shown:]
            ndcgs.append(ndcg_at_k(ranked, gt, TOP_K))
            recalls.append(recall_at_k(ranked, gt, TOP_K))
            in_tok += r.get("n_in", 0)
            out_tok += r.get("n_out", 0)
        if not ndcgs:
            continue
        n_m, n_lo, n_hi = bootstrap_ci(ndcgs)
        r_m, r_lo, r_hi = bootstrap_ci(recalls)
        p_in, p_out = PRICES.get(variant, (0, 0))
        cost = (in_tok * p_in + out_tok * p_out) / 1e6
        out["models"][variant] = {
            "n": len(ndcgs),
            "ndcg10_mean": n_m, "ndcg10_ci": [n_lo, n_hi],
            "recall10_mean": r_m, "recall10_ci": [r_lo, r_hi],
            "vs_cf_ndcg_pct": (n_m - cf_n_m) / (cf_n_m + 1e-9) * 100,
            "vs_cf_recall_pct": (r_m - cf_r_m) / (cf_r_m + 1e-9) * 100,
            "in_tokens": in_tok, "out_tokens": out_tok,
            "cost_usd": round(cost, 4),
            "avg_latency_s": round(float(np.mean([r.get("latency_s", 0) for r in recs.values()])), 2),
        }
        print(f"  {variant:<28} NDCG@10={n_m:.4f}  Recall@10={r_m:.4f}  "
              f"n={len(ndcgs)}  cost=${cost:.3f}")
    return out


def main():
    summary = {"config": {"seed": SEED, "K": K, "top_k": TOP_K, "n_factors": N_FACTORS}, "datasets": {}}
    for ds, path in DATASETS.items():
        try:
            summary["datasets"][ds] = aggregate_dataset(ds, path)
        except Exception as e:
            print(f"  {ds}: FAILED ({e})")

    out_path = PROJECT / "experiments/logs/v4_aggregate_n500.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")

    print("\n" + "="*78)
    print("FINAL TABLE — NDCG@10 (vs CF %) and Recall@10")
    print("="*78)
    for ds, d in summary["datasets"].items():
        cf = d["CF"]
        print(f"\n  {ds.upper()}  (n={d['n_users']}, CF-NDCG@10={cf['ndcg10_mean']:.4f}, "
              f"CF-Recall@10={cf['recall10_mean']:.4f}, ceiling={cf['recall100_mean']:.4f})")
        print(f"  {'Model':<28} {'NDCG@10':>10} {'vs CF':>9} {'Recall@10':>10} {'vs CF':>9} {'$':>7}")
        print(f"  {'-'*78}")
        for v, r in d["models"].items():
            print(f"  {v:<28} {r['ndcg10_mean']:>10.4f} {r['vs_cf_ndcg_pct']:>+8.1f}% "
                  f"{r['recall10_mean']:>10.4f} {r['vs_cf_recall_pct']:>+8.1f}% "
                  f"{r['cost_usd']:>7.3f}")


if __name__ == "__main__":
    main()
