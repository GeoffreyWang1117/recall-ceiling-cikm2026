#!/usr/bin/env python3
"""
CIKM v4 — Learned LambdaMART fusion with CF + Popularity + BM25 + Dense features.

Reviewer concern: industrial cascade rerankers learn from multiple signals
(CF score, popularity, BM25, text embeddings) rather than overwriting CF
with LLM-only re-rankings. Test whether a properly learned fusion under
disjoint user split can break the recall ceiling.

Feature set (12 features per candidate):
  1-8.  Existing build_features (cf score, rank, popularity, sim, history_len,
        cooc, 1/(rank+1), item_norm)
  9.    BM25 score (item title vs concatenated last-20 history titles)
  10.   Dense cosine (MiniLM, item emb vs avg of last-20 history embs)
  11.   BM25 rank (normalized)
  12.   Dense rank (normalized)

Protocol:
  - 250 train users disjoint from 500 eval users (Protocol C from leak fix).
  - K=100 CF candidates per user.
  - Label = 1 if candidate is the held-out target, else 0.
  - LambdaMART (LightGBM) with same hyperparameters as cikm_neural_rerankers_disjoint.
  - Evaluate on the 500 eval users; report NDCG@10 with paired Wilcoxon vs CF.

Output: experiments/logs/cikm_learned_fusion_v4.json
"""
from __future__ import annotations

import argparse, json, sys, re, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from cikm_controlled_recall_sweep import (
    CFSVD, build_features, LambdaMART,
    ndcg_at_k, safe_wilcoxon, bootstrap_ci,
)

DATA = REPO / "data" / "processed"
LOG = REPO / "experiments" / "logs"


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


_TOK = re.compile(r"[a-z0-9]+")
def tok(s: str): return _TOK.findall(s.lower())


def make_bm25(item_texts: Dict[int, str]):
    from rank_bm25 import BM25Okapi
    item_ids = sorted(item_texts.keys())
    docs = [tok(item_texts[i]) or ["_empty_"] for i in item_ids]
    bm25 = BM25Okapi(docs)
    return bm25, item_ids


def make_dense(item_texts: Dict[int, str]):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    item_ids = sorted(item_texts.keys())
    texts = [item_texts[i] or "(empty)" for i in item_ids]
    emb = model.encode(texts, batch_size=256, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    return emb, {i: idx for idx, i in enumerate(item_ids)}


def build_features_plus(uid: int, cands: List[int], cf_scores: List[float],
                        cf: CFSVD, user_history: Dict[int, List[int]],
                        item_pop: Dict[int, int], item_texts: Dict[int, str],
                        bm25, bm25_item_ids: List[int],
                        dense_emb: np.ndarray, dense_idx: Dict[int, int]) -> np.ndarray:
    base = build_features(uid, cands, cf_scores, cf, user_history, item_pop)
    hist = user_history.get(uid, [])[-20:]
    # BM25: query = concat history titles
    hist_txt = " ".join(item_texts.get(int(h), "") for h in hist)
    q_tokens = tok(hist_txt) or ["_empty_"]
    bm25_scores_all = bm25.get_scores(q_tokens)
    item_to_bm25idx = {bm25_item_ids[i]: i for i in range(len(bm25_item_ids))}
    bm25_scores = np.array([
        bm25_scores_all[item_to_bm25idx[c]] if c in item_to_bm25idx else 0.0
        for c in cands
    ])
    bm25_ranks = bm25_scores.argsort()[::-1].argsort().astype(float) / max(len(cands), 1)
    # Dense
    hist_idx = [dense_idx[h] for h in hist if h in dense_idx]
    if hist_idx:
        uvec = dense_emb[hist_idx].mean(axis=0)
        uvec = uvec / (np.linalg.norm(uvec) + 1e-9)
        dense_cand = np.array([
            float(dense_emb[dense_idx[c]] @ uvec) if c in dense_idx else 0.0
            for c in cands
        ])
    else:
        dense_cand = np.zeros(len(cands))
    dense_ranks = dense_cand.argsort()[::-1].argsort().astype(float) / max(len(cands), 1)
    extra = np.stack([bm25_scores, dense_cand, bm25_ranks, dense_ranks], axis=1).astype(np.float32)
    return np.concatenate([base, extra], axis=1)


def evaluate(ds: str, n_train: int, n_eval: int, K: int, seed: int = 42) -> Dict:
    print(f"\n=== {ds} ===")
    train_df = pd.read_parquet(DATA / ds / "train.parquet")
    val_df = pd.read_parquet(DATA / ds / "val.parquet") if (DATA / ds / "val.parquet").exists() else None
    test_df = pd.read_parquet(DATA / ds / "test.parquet")

    item_texts = load_item_texts(ds)
    print("  fitting CFSVD ...")
    cf = CFSVD(n_factors=128)
    cf.fit(train_df)
    print("  building BM25 ...")
    bm25, bm25_item_ids = make_bm25(item_texts)
    print("  encoding Dense ...")
    dense_emb, dense_idx = make_dense(item_texts)

    train_users_all = train_df["user_id"].drop_duplicates().tolist()
    test_users_all = test_df["user_id"].drop_duplicates().tolist()
    rng = np.random.RandomState(seed)
    eval_users = sorted(rng.choice(test_users_all, min(n_eval, len(test_users_all)), replace=False).tolist())
    avail_train = [u for u in train_users_all if u not in set(eval_users)]
    train_pool = sorted(rng.choice(avail_train, min(n_train, len(avail_train)), replace=False).tolist())
    print(f"  train_pool={len(train_pool)} eval_users={len(eval_users)} disjoint={len(set(train_pool)&set(eval_users))==0}")

    user_hist = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
    item_pop = train_df["item_id"].value_counts().to_dict()
    test_gt = test_df.set_index("user_id")["item_id"].to_dict()

    # === build training data: for each train user, retrieve K cands and label
    X_list, y_list, group_sizes = [], [], []
    for uid in tqdm(train_pool, desc=f"{ds} train-feat"):
        hist = user_hist.get(uid, [])
        if len(hist) < 3:
            continue
        # use uid's last interaction as positive label, exclude it from history
        train_label = hist[-1]
        hist_for_user = hist[:-1]
        # retrieve top-K from train pool excluding train_label
        exclude = set(hist_for_user)
        cands, scores = cf.topk(uid, K, exclude)
        # inject the label if not retrieved
        if train_label not in cands:
            cands.append(train_label); scores.append(0.0)
        # build features using user_hist - {train_label}
        ph = dict(user_hist); ph[uid] = hist_for_user
        X = build_features_plus(uid, cands, scores, cf, ph, item_pop,
                                item_texts, bm25, bm25_item_ids, dense_emb, dense_idx)
        y = np.array([1 if c == train_label else 0 for c in cands], dtype=int)
        X_list.append(X); y_list.append(y); group_sizes.append(len(cands))
    X_train = np.concatenate(X_list); y_train = np.concatenate(y_list)
    print(f"  train_features: X={X_train.shape}, n_users={len(group_sizes)}, pos_rate={y_train.mean():.4f}")

    # === train LambdaMART
    print("  training LambdaMART ...")
    lm = LambdaMART(n_leaves=31, n_estimators=100, lr=0.1)
    lm.fit(X_train, y_train, group_sizes)

    # === evaluate
    cf_ndcgs, lm_ndcgs = [], []
    for uid in tqdm(eval_users, desc=f"{ds} eval"):
        gt = test_gt.get(uid)
        if gt is None: continue
        hist = user_hist.get(uid, [])
        cands, scores = cf.topk(uid, K, set(hist))
        X = build_features_plus(uid, cands, scores, cf, user_hist, item_pop,
                                item_texts, bm25, bm25_item_ids, dense_emb, dense_idx)
        # CF baseline ranking is just by cf_score (already sorted in cands)
        cf_ndcgs.append(ndcg_at_k(cands, gt, k=10))
        # LambdaMART rerank
        pred = lm.predict(X)
        order = np.argsort(-pred)
        ranked = [cands[i] for i in order]
        lm_ndcgs.append(ndcg_at_k(ranked, gt, k=10))

    cf_arr = np.array(cf_ndcgs); lm_arr = np.array(lm_ndcgs)
    cf_mean, cf_lo, cf_hi = bootstrap_ci(cf_arr, n=2000)
    lm_mean, lm_lo, lm_hi = bootstrap_ci(lm_arr, n=2000)
    try:
        p = safe_wilcoxon(lm_arr, cf_arr)
        if isinstance(p, tuple): _, p = p
        if isinstance(p, float) and np.isnan(p): p = None
    except Exception:
        p = None
    return {
        "dataset": ds,
        "n_train": len(group_sizes),
        "n_eval": len(eval_users),
        "K": K,
        "n_features": 12,
        "feature_names": ["cf_score","cf_rank_norm","log_pop","cf_sim","log_hist_len",
                          "cf_cooc","inv_rank","item_norm","bm25_score","dense_score",
                          "bm25_rank_norm","dense_rank_norm"],
        "cf_ndcg10": {"mean": float(cf_mean), "ci": [float(cf_lo), float(cf_hi)]},
        "fusion_ndcg10": {"mean": float(lm_mean), "ci": [float(lm_lo), float(lm_hi)]},
        "delta_pct": float(100*(lm_mean - cf_mean)/max(cf_mean, 1e-9)),
        "p_paired_wilcoxon": float(p) if p is not None else None,
        "feature_importance": lm.model.feature_importance().tolist() if hasattr(lm, "model") else None,
        "per_user_cf": cf_arr.tolist(),
        "per_user_fusion": lm_arr.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["amazon_beauty_sampled","amazon_movies_sampled","amazon_electronics_sampled"])
    ap.add_argument("--n_train", type=int, default=250)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(LOG / "cikm_learned_fusion_v4.json"))
    args = ap.parse_args()
    out = {}
    for ds in args.datasets:
        try:
            out[ds] = evaluate(ds, args.n_train, args.n_eval, args.K, args.seed)
        except Exception as e:
            import traceback; traceback.print_exc()
            out[ds] = {"error": str(e)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n>>> wrote {args.out}")
    print("\n=== SUMMARY ===")
    print(f"{'Dataset':22s} {'CF NDCG':>10s} {'Fusion NDCG':>12s} {'Δ%':>7s} {'p':>8s}")
    for ds, r in out.items():
        if "cf_ndcg10" not in r: continue
        p_val = r.get('p_paired_wilcoxon')
        p_str = f"{p_val:.3f}" if isinstance(p_val, float) else "-"
        print(f"{ds:22s} {r['cf_ndcg10']['mean']:>10.4f} "
              f"{r['fusion_ndcg10']['mean']:>12.4f} {r['delta_pct']:>6.1f}% "
              f"{p_str:>8}")


if __name__ == "__main__":
    main()
