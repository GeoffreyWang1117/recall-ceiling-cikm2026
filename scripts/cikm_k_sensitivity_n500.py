#!/usr/bin/env python3
"""
CIKM 2026 P0-B: K-sensitivity at n=500 (cloud API for consistency)
==================================================================

Replaces the n=200 numbers in `tab:k_sensitivity` and `fig:recall_performance`
of the CIKM draft. The original `kdd_k_sensitivity.py` used local Qwen2.5-3B
under 4-bit quantization, which is inconsistent with the rest of the paper's
n=500 cloud-API multi-model harness.

This script uses gpt-4o-mini (the cheapest entry in `tab:models`) so the
K-sweep numbers are directly comparable to that table's K=100 row.

Datasets: Beauty (matches the paper's K-sensitivity narrative).
K values:  {10, 20, 50, 100, 200}.
n_users:   500, SEED=42.

Cost estimate: 500 users × 5 K-values × ~$0.0001 = ~$0.25 total.

Usage
-----
    python scripts/cikm_k_sensitivity_n500.py --n_users 500
    python scripts/cikm_k_sensitivity_n500.py --quick   # n=50 smoke test

Output
------
    experiments/logs/cikm_k_sensitivity_n500.json
    experiments/logs/checkpoints/cikm_ksens_K{K}_beauty.jsonl
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
from scipy.stats import wilcoxon
from tqdm import tqdm

import openai

from cikm_controlled_recall_sweep import CFSVD, ndcg_at_k, safe_wilcoxon, bootstrap_ci

# ---------------------------------------------------------------------------
# Config — mirrored from rebuttal_multiapi_n500.py for consistency
# ---------------------------------------------------------------------------
SEED = 42
TOP_K = 10
N_FACTORS = 128

OPENAI_BASE = "https://api.openai.com/v1"
MODEL = "gpt-4o-mini"
PRICE_IN, PRICE_OUT = 0.15, 0.60   # $/M tokens

PROMPT = (
    "Based on the user's purchase history, rank these candidate items by relevance.\n\n"
    "User's recent purchases:\n{history}\n\n"
    "Candidate items to rank:\n{candidates}\n\n"
    "Output only the top {top_k} most relevant item numbers in order (most relevant first). "
    "No explanation.\n"
    "Format: Item <number> on each line."
)


def get_api_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        env_file = project_root / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found")
    return key


def call_llm(client, prompt, max_tokens=120, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.1, timeout=30,
            )
            text = r.choices[0].message.content or ""
            return text, r.usage.prompt_tokens, r.usage.completion_tokens
        except openai.RateLimitError:
            time.sleep(2 ** attempt + np.random.uniform(0, 1))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "", 0, 0
    return "", 0, 0


def parse_ranking(response, n_cands):
    nums = re.findall(r"\d+", response)
    seen, out = set(), []
    for x in nums:
        i = int(x) - 1
        if 0 <= i < n_cands and i not in seen:
            out.append(i); seen.add(i)
    for i in range(n_cands):
        if i not in seen:
            out.append(i)
    return out[:n_cands]


# ---------------------------------------------------------------------------
# Main K-sweep
# ---------------------------------------------------------------------------

def run_for_K(K, cf, test_samples, item_texts, client, ckpt_path):
    """Return list of per-user records: dict(user_id, recall, ndcg_cf, ndcg_llm, n_in, n_out)."""
    done = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done[str(rec["user_id"])] = rec
            except Exception:
                continue
    print(f"  K={K}: {len(done)}/{len(test_samples)} users already in checkpoint")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(ckpt_path, "a")

    records = list(done.values())
    for sample in tqdm(test_samples, desc=f"K={K}", leave=False):
        uid = str(sample["user_id"])
        if uid in done:
            continue
        try:
            gt = sample["ground_truth"]
            history = sample["history"]
            cands, _ = cf.topk(int(uid), K, history)
            if not cands:
                rec = dict(user_id=uid, recall=0.0, ndcg_cf=0.0, ndcg_llm=0.0,
                           n_in=0, n_out=0, latency_s=0.0)
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                records.append(rec); continue

            recall = len(set(cands) & set(gt)) / max(len(gt), 1)
            ndcg_cf = ndcg_at_k(cands, gt, k=10)

            # LLM rerank
            hist_text = ", ".join([item_texts.get(int(h), f"Item {h}") for h in history[-5:]])
            cand_text = "\n".join([f"{i+1}. {item_texts.get(int(c), f'Item {c}')}"
                                    for i, c in enumerate(cands)])
            prompt = PROMPT.format(history=hist_text, candidates=cand_text, top_k=TOP_K)

            t0 = time.time()
            resp, n_in, n_out = call_llm(client, prompt)
            latency = time.time() - t0

            order = parse_ranking(resp, len(cands))
            llm_ranked = [cands[i] for i in order]
            ndcg_llm = ndcg_at_k(llm_ranked, gt, k=10)

            rec = dict(user_id=uid, recall=recall, ndcg_cf=ndcg_cf, ndcg_llm=ndcg_llm,
                       n_in=n_in, n_out=n_out, latency_s=latency)
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            records.append(rec)
        except Exception as e:
            print(f"\n  user {uid} failed: {str(e)[:100]}")
            continue
    fout.close()
    return records


def summarise(records):
    recalls = [r["recall"] for r in records]
    ndcg_cf = [r["ndcg_cf"] for r in records]
    ndcg_llm = [r["ndcg_llm"] for r in records]
    n_in = sum(r["n_in"] for r in records)
    n_out = sum(r["n_out"] for r in records)

    cf_mean, cf_lo, cf_hi = bootstrap_ci(ndcg_cf, seed=0)
    llm_mean, llm_lo, llm_hi = bootstrap_ci(ndcg_llm, seed=1)
    p = safe_wilcoxon(ndcg_llm, ndcg_cf)
    cost_usd = n_in * PRICE_IN / 1e6 + n_out * PRICE_OUT / 1e6
    return dict(
        n=len(records),
        recall_mean=float(np.mean(recalls)),
        cf_ndcg=cf_mean, cf_ci=[cf_lo, cf_hi],
        llm_ndcg=llm_mean, llm_ci=[llm_lo, llm_hi],
        delta=llm_mean - cf_mean,
        improvement_pct=(llm_mean - cf_mean) / cf_mean * 100 if cf_mean > 0 else 0.0,
        wilcoxon_p=p,
        tokens_in=n_in, tokens_out=n_out,
        cost_usd=cost_usd,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=500)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dataset", default="beauty")
    ap.add_argument("--K_values", default="10,20,50,100,200")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="experiments/logs/cikm_k_sensitivity_n500.json")
    args = ap.parse_args()

    if args.quick:
        args.n_users = 50; args.K_values = "50,100"

    K_values = [int(k) for k in args.K_values.split(",")]
    print(f"K values: {K_values}  n_users={args.n_users}  dataset={args.dataset}")

    # Load dataset
    ds_path = project_root / "data" / "processed" / f"amazon_{args.dataset}_sampled"
    train_df = pd.read_parquet(ds_path / "train.parquet").astype({"user_id": int, "item_id": int})
    test_df = pd.read_parquet(ds_path / "test.parquet").astype({"user_id": int, "item_id": int})

    # Item titles
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

    # Build CF
    cf = CFSVD(n_factors=N_FACTORS); cf.fit(train_df)
    user_history = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
    test_gt = test_df.groupby("user_id")["item_id"].apply(list).to_dict()

    rng = np.random.default_rng(args.seed)
    eligible = [u for u in test_gt if u in user_history and len(user_history[u]) >= 3]
    n = min(args.n_users, len(eligible))
    sampled = list(rng.choice(eligible, size=n, replace=False))
    test_samples = [dict(user_id=int(u), history=user_history[u],
                         ground_truth=test_gt[u]) for u in sampled]
    print(f"Test samples: {len(test_samples)}")

    # Client
    client = openai.OpenAI(api_key=get_api_key(), base_url=OPENAI_BASE)

    # Run each K
    t0 = time.time()
    results_by_K = {}
    for K in K_values:
        ckpt = project_root / "experiments" / "logs" / "checkpoints" / f"cikm_ksens_K{K}_{args.dataset}.jsonl"
        records = run_for_K(K, cf, test_samples, item_texts, client, ckpt)
        s = summarise(records)
        results_by_K[str(K)] = s
        print(f"  K={K:>3}: recall={s['recall_mean']*100:5.2f}%  "
              f"CF={s['cf_ndcg']:.4f}  LLM={s['llm_ndcg']:.4f}  "
              f"Δ={(s['llm_ndcg']-s['cf_ndcg']):+.4f}  "
              f"({s['improvement_pct']:+.1f}%, p={s['wilcoxon_p']})  ${s['cost_usd']:.3f}")

    total_cost = sum(s["cost_usd"] for s in results_by_K.values())
    summary = dict(
        experiment="cikm_k_sensitivity_n500",
        dataset=args.dataset,
        model=MODEL,
        K_values=K_values,
        n_users=args.n_users,
        seed=args.seed,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        results=results_by_K,
        total_cost_usd=total_cost,
        elapsed_sec=time.time() - t0,
    )
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\nTotal cost: ${total_cost:.3f}  elapsed: {summary['elapsed_sec']:.0f}s")
    print(f"Saved: {out_path}")

    # Pretty final table
    print(f"\n{'K':>4} {'Recall%':>8} {'CF NDCG':>10} {'LLM NDCG':>10} {'Δ':>10} {'LLM vs CF':>12} {'p':>8}")
    for K in K_values:
        s = results_by_K[str(K)]
        print(f"{K:>4} {s['recall_mean']*100:>7.2f}% {s['cf_ndcg']:>10.4f} "
              f"{s['llm_ndcg']:>10.4f} {s['delta']:>+10.4f} "
              f"{s['improvement_pct']:>+10.1f}% {s['wilcoxon_p']:>8.4f}")


if __name__ == "__main__":
    main()
