#!/usr/bin/env python3
"""
CIKM 2026 P1-B: Adaptive-K End-to-End NDCG
===========================================

Adaptive-K (RAEP Step 4) is currently reported in `tab:raep` only as a
recall improvement: +62–105% Recall@K across datasets. Reviewers (R2/R4
round 1) flagged that the paper's central thesis is "recall does not
translate to NDCG under LLM reranking" — yet `tab:raep` reports only
recall, leaving open the internal-consistency question: does the recall
gain from adaptive-K translate to NDCG when an LLM reranker is layered
on top?

This script measures the end-to-end pipeline:
    adaptive-K retrieval  →  gpt-4o-mini reranking  →  NDCG@10

vs. the fixed-K=100 baseline at the same user set.

Datasets: Beauty, Movies, Electronics. n=500 per dataset.
Model: gpt-4o-mini (consistent with `tab:models` cheapest entry).

Expected outcome (per the ceiling thesis): adaptive-K raises recall but
NDCG@10 under LLM rerank does not significantly improve, because the
ceiling binds across the recall range we can reach via adaptive-K.

Cost: ~1500 × 1 call × ~$0.0003 (longer adaptive prompts) ≈ $0.45.

Usage
-----
    python scripts/cikm_adaptive_k_ndcg.py --n_users 500
    python scripts/cikm_adaptive_k_ndcg.py --quick

Output
------
    experiments/logs/cikm_adaptive_k_ndcg.json
    experiments/logs/checkpoints/cikm_adaptiveK_{dataset}.jsonl
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

import argparse
import json
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import openai

from cikm_controlled_recall_sweep import CFSVD, ndcg_at_k, safe_wilcoxon, bootstrap_ci
from cikm_k_sensitivity_n500 import get_api_key, call_llm, parse_ranking, PROMPT, MODEL, PRICE_IN, PRICE_OUT


# ---------------------------------------------------------------------------
# Adaptive K: pick K per user from CF score concentration
# (re-implements the simple difficulty heuristic from kdd_recall_aware_evaluation.py
#  but on top of our CFSVD class)
# ---------------------------------------------------------------------------

def cf_all_scores(cf, uid, exclude_ids):
    """Return dict item_id -> score for all items."""
    if uid not in cf.user_to_idx:
        return {}
    u = cf.user_factors[cf.user_to_idx[uid]]
    s = cf.item_factors @ u
    for it in exclude_ids:
        if it in cf.item_to_idx:
            s[cf.item_to_idx[it]] = -np.inf
    return s


def adaptive_K(cf, uid, exclude_ids, K_min=50, K_max=500):
    s = cf_all_scores(cf, uid, exclude_ids)
    if len(s) == 0:
        return K_max, 1.0
    valid = s[s > -np.inf]
    if len(valid) < 100:
        return K_max, 1.0
    valid_sorted = np.sort(valid)[::-1]
    top100 = float(np.mean(valid_sorted[:100]))
    rest = float(np.mean(valid_sorted[100:min(500, len(valid_sorted))]))
    conc = top100 / (rest + 1e-10) if rest > 0 else 1.0
    difficulty = 1.0 / (1.0 + np.log1p(conc))
    K = int(K_min + difficulty * (K_max - K_min))
    K = max(K_min, min(K_max, K))
    return K, difficulty


def topk_with_K(cf, uid, K, exclude_ids):
    s = cf_all_scores(cf, uid, exclude_ids)
    if len(s) == 0:
        return [], []
    top = np.argsort(s)[::-1][:K]
    cands = [cf.idx_to_item[i] for i in top if s[i] > -np.inf]
    scores = [float(s[cf.item_to_idx[c]]) for c in cands]
    return cands, scores


# ---------------------------------------------------------------------------
# Per-dataset run
# ---------------------------------------------------------------------------

def run_dataset(ds_name, args, client, item_texts_cache=None):
    print(f"\n{'='*72}\n{ds_name.upper()}\n{'='*72}")
    ds_path = project_root / "data" / "processed" / f"amazon_{ds_name}_sampled"
    train_df = pd.read_parquet(ds_path / "train.parquet").astype({"user_id": int, "item_id": int})
    test_df = pd.read_parquet(ds_path / "test.parquet").astype({"user_id": int, "item_id": int})

    # item texts
    item_texts = {}
    meta_path = ds_path / "item_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for k, v in meta.items():
            try:
                k = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                t = v.get("title") or v.get("text") or f"Item {k}"
                item_texts[k] = str(t)[:200]
            else:
                item_texts[k] = f"Item {k}"

    cf = CFSVD(n_factors=128); cf.fit(train_df)
    user_history = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
    test_gt = test_df.groupby("user_id")["item_id"].apply(list).to_dict()

    rng = np.random.default_rng(args.seed)
    eligible = [u for u in test_gt if u in user_history and len(user_history[u]) >= 3]
    n = min(args.n_users, len(eligible))
    sampled = list(rng.choice(eligible, size=n, replace=False))
    print(f"  eligible={len(eligible)}  sampled={len(sampled)}")

    # Checkpoint
    ckpt = project_root / "experiments" / "logs" / "checkpoints" / f"cikm_adaptiveK_{ds_name}.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line); done[str(rec["user_id"])] = rec
                except Exception:
                    continue
    print(f"  checkpoint: {len(done)}/{len(sampled)} users done")

    fout = open(ckpt, "a")
    records = list(done.values())

    for uid in tqdm(sampled, desc=f"  {ds_name}"):
        uid = int(uid)
        if str(uid) in done:
            continue
        try:
            gt = test_gt[uid]
            hist = user_history[uid]

            # Fixed K=100 baseline
            fk_cands, _ = topk_with_K(cf, uid, 100, hist)
            fk_recall = len(set(fk_cands) & set(gt)) / max(len(gt), 1)
            fk_ndcg_cf = ndcg_at_k(fk_cands, gt, k=10)

            # Adaptive-K
            K_adapt, difficulty = adaptive_K(cf, uid, hist)
            ak_cands, _ = topk_with_K(cf, uid, K_adapt, hist)
            ak_recall = len(set(ak_cands) & set(gt)) / max(len(gt), 1)
            ak_ndcg_cf = ndcg_at_k(ak_cands, gt, k=10)

            # LLM rerank for both — top-100 budget for LLM input
            # (otherwise adaptive K=500 makes prompts unreasonably long; we use
            #  the adaptive-K-derived candidates truncated to 100 for fairness with the K=100 baseline,
            #  but the candidate POOL is from adaptive K, so coverage differs).
            ak_for_llm = ak_cands[:100] if len(ak_cands) > 100 else ak_cands
            fk_for_llm = fk_cands[:100]

            def llm_rerank(cands):
                if not cands:
                    return [], 0, 0, 0.0
                hist_text = ", ".join([item_texts.get(int(h), f"Item {h}") for h in hist[-5:]])
                cand_text = "\n".join([f"{i+1}. {item_texts.get(int(c), f'Item {c}')}"
                                        for i, c in enumerate(cands)])
                p = PROMPT.format(history=hist_text, candidates=cand_text, top_k=10)
                t0 = time.time()
                resp, n_in, n_out = call_llm(client, p)
                latency = time.time() - t0
                order = parse_ranking(resp, len(cands))
                ranked = [cands[i] for i in order]
                return ranked, n_in, n_out, latency

            fk_ranked, fk_in, fk_out, fk_lat = llm_rerank(fk_for_llm)
            ak_ranked, ak_in, ak_out, ak_lat = llm_rerank(ak_for_llm)
            fk_ndcg_llm = ndcg_at_k(fk_ranked, gt, k=10)
            ak_ndcg_llm = ndcg_at_k(ak_ranked, gt, k=10)

            rec = dict(
                user_id=uid,
                K_adapt=K_adapt, difficulty=float(difficulty),
                fk_recall=fk_recall, fk_ndcg_cf=fk_ndcg_cf, fk_ndcg_llm=fk_ndcg_llm,
                ak_recall=ak_recall, ak_ndcg_cf=ak_ndcg_cf, ak_ndcg_llm=ak_ndcg_llm,
                tokens_in=int(fk_in + ak_in), tokens_out=int(fk_out + ak_out),
                latency_s=float(fk_lat + ak_lat),
            )
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            records.append(rec)
        except Exception as e:
            print(f"\n  user {uid} error: {str(e)[:100]}")
            continue
    fout.close()

    # Aggregate
    fk_recall = [r["fk_recall"] for r in records]
    ak_recall = [r["ak_recall"] for r in records]
    fk_ndcg_cf = [r["fk_ndcg_cf"] for r in records]
    ak_ndcg_cf = [r["ak_ndcg_cf"] for r in records]
    fk_ndcg_llm = [r["fk_ndcg_llm"] for r in records]
    ak_ndcg_llm = [r["ak_ndcg_llm"] for r in records]
    K_used = [r["K_adapt"] for r in records]
    n_in = sum(r["tokens_in"] for r in records)
    n_out = sum(r["tokens_out"] for r in records)
    cost = n_in * PRICE_IN / 1e6 + n_out * PRICE_OUT / 1e6

    def stats(vals, seed):
        m, lo, hi = bootstrap_ci(vals, seed=seed)
        return dict(mean=m, ci=[lo, hi])

    out = dict(
        dataset=ds_name,
        n=len(records),
        K_adaptive=dict(mean=float(np.mean(K_used)),
                        median=float(np.median(K_used)),
                        min=int(min(K_used)) if K_used else 0,
                        max=int(max(K_used)) if K_used else 0),
        recall=dict(
            fixed_K100=stats(fk_recall, 0),
            adaptive_K=stats(ak_recall, 1),
            improvement_pct=(np.mean(ak_recall) - np.mean(fk_recall))
                            / np.mean(fk_recall) * 100 if np.mean(fk_recall) > 0 else 0.0,
        ),
        ndcg_cf=dict(
            fixed_K100=stats(fk_ndcg_cf, 2),
            adaptive_K=stats(ak_ndcg_cf, 3),
            wilcoxon_p=safe_wilcoxon(ak_ndcg_cf, fk_ndcg_cf),
        ),
        ndcg_llm=dict(
            fixed_K100=stats(fk_ndcg_llm, 4),
            adaptive_K=stats(ak_ndcg_llm, 5),
            wilcoxon_p=safe_wilcoxon(ak_ndcg_llm, fk_ndcg_llm),
            llm_vs_cf_fixed=safe_wilcoxon(fk_ndcg_llm, fk_ndcg_cf),
            llm_vs_cf_adapt=safe_wilcoxon(ak_ndcg_llm, ak_ndcg_cf),
        ),
        tokens=dict(input=n_in, output=n_out),
        cost_usd=cost,
    )
    print(f"  K_adapt mean={out['K_adaptive']['mean']:.0f}  "
          f"recall fixed={np.mean(fk_recall)*100:.2f}% → adapt={np.mean(ak_recall)*100:.2f}% "
          f"({out['recall']['improvement_pct']:+.1f}%)")
    print(f"  NDCG_CF  fixed={np.mean(fk_ndcg_cf):.4f}  adapt={np.mean(ak_ndcg_cf):.4f}  "
          f"p={out['ndcg_cf']['wilcoxon_p']}")
    print(f"  NDCG_LLM fixed={np.mean(fk_ndcg_llm):.4f}  adapt={np.mean(ak_ndcg_llm):.4f}  "
          f"p={out['ndcg_llm']['wilcoxon_p']}")
    print(f"  LLM-vs-CF (fixed)={out['ndcg_llm']['llm_vs_cf_fixed']}  "
          f"(adapt)={out['ndcg_llm']['llm_vs_cf_adapt']}")
    print(f"  cost=${cost:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", default="beauty,movies,electronics")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="experiments/logs/cikm_adaptive_k_ndcg.json")
    args = ap.parse_args()

    if args.quick:
        args.n_users = 30; args.datasets = "beauty"

    client = openai.OpenAI(api_key=get_api_key(),
                            base_url="https://api.openai.com/v1")

    t0 = time.time()
    datasets = [d.strip() for d in args.datasets.split(",")]
    by_dataset = {}
    for ds in datasets:
        by_dataset[ds] = run_dataset(ds, args, client)

    summary = dict(
        experiment="cikm_adaptive_k_ndcg_P1B",
        model=MODEL,
        n_users_target=args.n_users,
        seed=args.seed,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        results=by_dataset,
        total_cost_usd=sum(r["cost_usd"] for r in by_dataset.values()),
        elapsed_sec=time.time() - t0,
    )
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*72}\nSUMMARY — Adaptive-K end-to-end\n{'='*72}")
    print(f"{'Dataset':<14}{'recall_fix':>11}{'recall_adapt':>13}{'Δ%':>8} "
          f"{'NDCG_LLM_fix':>14}{'NDCG_LLM_adapt':>16}{'p':>8}")
    for ds, r in by_dataset.items():
        print(f"{ds:<14}{r['recall']['fixed_K100']['mean']*100:>10.2f}%"
              f"{r['recall']['adaptive_K']['mean']*100:>12.2f}%"
              f"{r['recall']['improvement_pct']:>+7.1f}%"
              f"{r['ndcg_llm']['fixed_K100']['mean']:>14.4f}"
              f"{r['ndcg_llm']['adaptive_K']['mean']:>16.4f}"
              f"{r['ndcg_llm']['wilcoxon_p']:>8.4f}")
    print(f"\nTotal cost: ${summary['total_cost_usd']:.3f}  elapsed: {summary['elapsed_sec']:.0f}s")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
