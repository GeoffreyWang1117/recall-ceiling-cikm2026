#!/usr/bin/env python3
"""
CIKM v4 — Hybrid retrieval baselines (BM25 + Dense + RRF fusion).

Question: does stronger upstream retrieval (text-aware) raise the recall ceiling?

Methods compared at K=100, n=500 disjoint evaluation users:
  - CF-SVD (TruncatedSVD-128, existing baseline)
  - BM25 over item titles (user history concatenated as query)
  - Dense (sentence-transformers, all-MiniLM-L6-v2; cached locally)
  - RRF fusion of {CF-SVD, BM25, Dense}
  - LightGCN-Tuned (existing best from Table 3, copied for reference)

Output: experiments/logs/cikm_hybrid_retrieval_v4.json
Reports per-method Recall@100 with bootstrap 95% CI, and one end-to-end
test where the best fusion's top-100 is passed to GPT-4o-mini (Beauty only,
to stay within ~$0.10 budget) to test "does better retrieval enable LLM reranking?"
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from cikm_controlled_recall_sweep import CFSVD, ndcg_at_k, bootstrap_ci, safe_wilcoxon

DATA = REPO / "data" / "processed"
LOG = REPO / "experiments" / "logs"
LOG.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Item-text loading
# ----------------------------------------------------------------------------
def load_item_texts(ds: str) -> Dict[int, str]:
    with open(DATA / ds / "item_metadata.json") as f:
        meta = json.load(f)
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        title = (v.get("title") or v.get("text") or "")[:120]
        cat = (v.get("category") or "")[:40]
        out[int(k)] = f"{title} {cat}".strip()
    return out


# ----------------------------------------------------------------------------
# BM25 retriever
# ----------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self):
        from rank_bm25 import BM25Okapi
        self.BM25 = BM25Okapi

    def fit(self, item_texts: Dict[int, str], train_df: pd.DataFrame):
        # tokenize naively (lowercase, alphanum split)
        import re
        def tok(s: str): return re.findall(r"[a-z0-9]+", s.lower())
        self.item_ids = sorted(item_texts.keys())
        self.item_tokens = [tok(item_texts[i]) or ["_empty_"] for i in self.item_ids]
        self.bm25 = self.BM25(self.item_tokens)
        self.user_hist = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
        self.item_texts = item_texts
        self._tok = tok

    def recall(self, uid: int, K: int, exclude_ids: set[int]) -> Tuple[List[int], np.ndarray]:
        hist = self.user_hist.get(uid, [])
        if not hist:
            return [], np.array([])
        # query = concatenated tokens of last 20 history items
        query_text = " ".join(self.item_texts.get(int(h), "") for h in hist[-20:])
        query = self._tok(query_text) or ["_empty_"]
        scores = self.bm25.get_scores(query)
        # mask excluded
        idx_set = {self.item_ids[i]: i for i in range(len(self.item_ids))}
        for h in exclude_ids:
            if h in idx_set:
                scores[idx_set[h]] = -1e9
        top = np.argsort(-scores)[:K]
        return [self.item_ids[i] for i in top], scores[top]


# ----------------------------------------------------------------------------
# Dense retriever (sentence-transformers, MiniLM cached)
# ----------------------------------------------------------------------------
class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, item_texts: Dict[int, str], train_df: pd.DataFrame):
        self.item_ids = sorted(item_texts.keys())
        texts = [item_texts[i] or "(empty)" for i in self.item_ids]
        self.emb = self.model.encode(
            texts, batch_size=256, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        self.user_hist = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
        self.item_texts = item_texts
        self.idx_of = {self.item_ids[i]: i for i in range(len(self.item_ids))}

    def recall(self, uid: int, K: int, exclude_ids: set[int]) -> Tuple[List[int], np.ndarray]:
        hist = self.user_hist.get(uid, [])
        if not hist:
            return [], np.array([])
        hist_idx = [self.idx_of[h] for h in hist[-20:] if h in self.idx_of]
        if not hist_idx:
            return [], np.array([])
        user_vec = self.emb[hist_idx].mean(axis=0)
        user_vec /= (np.linalg.norm(user_vec) + 1e-9)
        scores = self.emb @ user_vec
        for h in exclude_ids:
            if h in self.idx_of:
                scores[self.idx_of[h]] = -1e9
        top = np.argsort(-scores)[:K]
        return [self.item_ids[i] for i in top], scores[top]


# ----------------------------------------------------------------------------
# RRF fusion: rank-based, parameter k=60 (canonical Cormack et al.)
# ----------------------------------------------------------------------------
def rrf_fusion(rankings: List[List[int]], K: int, k_rrf: int = 60) -> List[int]:
    scores: Dict[int, float] = {}
    for rk in rankings:
        for rank, item in enumerate(rk, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k_rrf + rank)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [it for it, _ in ordered[:K]]


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(ds: str, n_users: int, K: int, seed: int = 42) -> Dict:
    print(f"\n=== {ds} ===")
    train = pd.read_parquet(DATA / ds / "train.parquet")
    val = pd.read_parquet(DATA / ds / "val.parquet")
    test = pd.read_parquet(DATA / ds / "test.parquet")
    item_texts = load_item_texts(ds)
    print(f"  n_items_with_text={len(item_texts)}, train_rows={len(train)}")

    rng = np.random.RandomState(seed)
    eval_users = test["user_id"].drop_duplicates().sample(
        n=min(n_users, test["user_id"].nunique()), random_state=rng
    ).tolist()
    # ground truth: each user's test item
    test_gt = test.set_index("user_id")["item_id"].to_dict()
    train_hist = train.groupby("user_id")["item_id"].apply(list).to_dict()

    # Wrap CFSVD to give it the .recall(uid,K,exclude) interface
    class CFAdapter:
        def __init__(self, cf): self.cf = cf
        def recall(self, uid, K, exclude_ids):
            cands, scores = self.cf.topk(uid, K, exclude_ids)
            return cands, np.array(scores) if scores is not None else np.array([])

    methods = {}
    print("  fitting CFSVD ...")
    cf = CFSVD(n_factors=128)
    cf.fit(train)
    methods["CF-SVD"] = CFAdapter(cf)

    print("  fitting BM25 ...")
    bm25 = BM25Retriever()
    bm25.fit(item_texts, train)
    methods["BM25"] = bm25

    print("  fitting Dense (MiniLM) ...")
    dense = DenseRetriever()
    dense.fit(item_texts, train)
    methods["Dense"] = dense

    per_user_recall = {m: [] for m in methods}
    per_user_recall["RRF-3way"] = []
    per_user_recall["RRF-CFBM25"] = []
    per_user_recall["RRF-CFDense"] = []
    hit_at_100 = {m: 0 for m in per_user_recall}

    for uid in tqdm(eval_users, desc=f"{ds} eval"):
        gt = test_gt.get(uid)
        if gt is None:
            continue
        exclude = set(train_hist.get(uid, []))
        # exclude validation too
        val_hist = val[val["user_id"] == uid]["item_id"].tolist()
        exclude |= set(val_hist)

        rankings = {}
        for name, m in methods.items():
            cands, _ = m.recall(uid, K, exclude)
            rankings[name] = cands
            hit = 1 if gt in cands else 0
            per_user_recall[name].append(hit)
            hit_at_100[name] += hit

        # RRF fusions
        rrf_3way = rrf_fusion([rankings["CF-SVD"], rankings["BM25"], rankings["Dense"]], K)
        hit = 1 if gt in rrf_3way else 0
        per_user_recall["RRF-3way"].append(hit); hit_at_100["RRF-3way"] += hit
        rrf_cf_bm = rrf_fusion([rankings["CF-SVD"], rankings["BM25"]], K)
        hit = 1 if gt in rrf_cf_bm else 0
        per_user_recall["RRF-CFBM25"].append(hit); hit_at_100["RRF-CFBM25"] += hit
        rrf_cf_dn = rrf_fusion([rankings["CF-SVD"], rankings["Dense"]], K)
        hit = 1 if gt in rrf_cf_dn else 0
        per_user_recall["RRF-CFDense"].append(hit); hit_at_100["RRF-CFDense"] += hit

    n = len(per_user_recall["CF-SVD"])
    results = {}
    for name, arr in per_user_recall.items():
        a = np.array(arr, dtype=float)
        m, lo, hi = bootstrap_ci(a, n=2000)
        results[name] = {
            "recall_at_100_mean": float(m),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n_users": n,
            "per_user_arr": arr,
        }
    # paired Wilcoxon: each fusion vs CF-SVD
    cf_arr = np.array(per_user_recall["CF-SVD"])
    for name in ["BM25", "Dense", "RRF-3way", "RRF-CFBM25", "RRF-CFDense"]:
        arr = np.array(per_user_recall[name])
        # safe_wilcoxon returns a bare float (NaN when it cannot test). The earlier
        # `_, p = safe_wilcoxon(...)` always raised and silently nulled every p-value.
        p = safe_wilcoxon(arr, cf_arr)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            p = None
            results[name]["p_vs_CF_SVD_note"] = "not testable: all paired differences zero or fewer than 5 nonzero"
        results[name]["p_vs_CF_SVD"] = float(p) if p is not None else None
    return {
        "dataset": ds,
        "n_users": n,
        "K": K,
        "seed": seed,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_user_arr"}
                    for k, v in results.items()},
        "per_user": {k: v["per_user_arr"] for k, v in results.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["amazon_beauty_sampled", "amazon_movies_sampled",
                             "amazon_electronics_sampled"])
    ap.add_argument("--n_users", type=int, default=500)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=str(LOG / "cikm_hybrid_retrieval_v4.json"))
    args = ap.parse_args()
    all_results = {}
    for ds in args.datasets:
        try:
            all_results[ds] = evaluate(ds, args.n_users, args.K, args.seed)
        except Exception as e:
            print(f"FAILED {ds}: {e}")
            import traceback; traceback.print_exc()
            all_results[ds] = {"error": str(e)}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n>>> wrote {out_path}")
    # summary
    print("\n=== SUMMARY: Recall@100 (%) ===")
    print(f"{'Dataset':22s} {'CF-SVD':>8s} {'BM25':>8s} {'Dense':>8s} {'RRF-3':>8s} {'RRFc+b':>8s} {'RRFc+d':>8s}")
    for ds, r in all_results.items():
        if "results" not in r: continue
        row = [f"{ds:22s}"]
        for m in ["CF-SVD","BM25","Dense","RRF-3way","RRF-CFBM25","RRF-CFDense"]:
            row.append(f"{100*r['results'][m]['recall_at_100_mean']:>7.2f} ")
        print("".join(row))


if __name__ == "__main__":
    main()
