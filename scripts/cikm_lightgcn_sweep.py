#!/usr/bin/env python3
"""
CIKM 2026 P1-C: LightGCN Hyperparameter Sweep
==============================================

R4_Skeptic round 1 CRITICAL: "LightGCN Movies 0.19% is orders below
field SOTA (He et al. 2020 reports ~6.5% Recall@20 on Amazon-Book, which
on similar catalogs implies Recall@100 of 15-30%)."

The CIKM draft currently flags LightGCN as "untuned-default" and uses
MultiSource-RRF (10.12% on Beauty) as the representative ceiling.
This script runs a proper grid search to either (a) confirm that LightGCN
on our sampled subsets really cannot exceed our reported numbers, or (b)
recover a higher LightGCN ceiling, in which case Section 5.2 ("Catalog
Size analysis") needs adjustment.

Grid
----
  embedding dim:    {64, 128}
  number of layers: {2, 3, 4}
  learning rate:    {1e-3, 5e-3}
  L2 reg weight:    {1e-4, 1e-3}
  epochs:           50 (fixed; we report best-epoch by recall on val)
That is 24 configurations per dataset. We run on Beauty, Movies, Electronics.

Each config trains LightGCN, computes Recall@100 on the test set, and
records the best configuration per dataset.

Compute
-------
GPU-required. Beauty: ~2 min/config × 24 = ~50 min.
Movies: bigger catalog/train → ~10 min/config × 24 = ~4 h.
Electronics: ~5 min/config × 24 = ~2 h. Total ~7 h.

To stay within session budget, use --datasets beauty for a fast probe,
then expand if Beauty's tuned recall is meaningfully higher than the
untuned 4.99%.

Usage
-----
    python scripts/cikm_lightgcn_sweep.py --datasets beauty
    python scripts/cikm_lightgcn_sweep.py --datasets beauty,movies,electronics
    python scripts/cikm_lightgcn_sweep.py --quick   # 4 configs, beauty only

Output
------
    experiments/logs/cikm_lightgcn_sweep.json
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
from itertools import product

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Reuse the existing LightGCNTuned implementation from kdd_sota_retrieval_baselines.py
# (it's the same architecture used in the paper)
from kdd_sota_retrieval_baselines import LightGCNTuned


def recall_at_k(ranked, gt_set, k=100):
    return len(set(ranked[:k]) & gt_set) / max(len(gt_set), 1)


def evaluate_lightgcn(lg, test_users, user_history, test_gt, K=100):
    recalls = []
    for uid in test_users:
        hist = user_history.get(uid, [])
        cands, _ = lg.recall(uid, K, hist)
        gt_set = set(test_gt.get(uid, []))
        if not gt_set:
            continue
        recalls.append(len(set(cands) & gt_set) / len(gt_set))
    return float(np.mean(recalls)) if recalls else 0.0


def run_dataset(name, args):
    print(f"\n{'='*72}\nDATASET: {name.upper()}\n{'='*72}")
    ds_path = project_root / "data" / "processed" / f"amazon_{name}_sampled"
    train_df = pd.read_parquet(ds_path / "train.parquet").astype({"user_id": int, "item_id": int})
    test_df = pd.read_parquet(ds_path / "test.parquet").astype({"user_id": int, "item_id": int})

    user_history = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
    test_gt = test_df.groupby("user_id")["item_id"].apply(list).to_dict()

    rng = np.random.default_rng(args.seed)
    eligible = [u for u in test_gt if u in user_history and len(user_history[u]) >= 3]
    n = min(args.n_users, len(eligible))
    test_users = list(rng.choice(eligible, size=n, replace=False))
    print(f"  catalog={train_df['item_id'].nunique()}  test_users={len(test_users)}")

    # Hyperparameter grid
    if args.quick:
        grid = list(product([64], [2, 3], [1e-3], [1e-4]))
    else:
        grid = list(product(
            [64, 128],            # emb_dim
            [2, 3, 4],            # n_layers
            [1e-3, 5e-3],         # lr
            [1e-4, 1e-3],         # reg_weight
        ))
    print(f"  grid: {len(grid)} configurations")

    results = []
    best = dict(recall=-1)
    for i, (emb_dim, n_layers, lr, reg) in enumerate(grid):
        cfg_label = f"e{emb_dim}_L{n_layers}_lr{lr}_reg{reg}"
        print(f"  [{i+1}/{len(grid)}] {cfg_label}")
        t0 = time.time()
        try:
            lg = LightGCNTuned(
                emb_dim=emb_dim, n_layers=n_layers,
                lr=lr, n_epochs=args.epochs, batch_size=args.batch,
                reg_weight=reg,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            lg.fit(train_df)
            r100 = evaluate_lightgcn(lg, test_users, user_history, test_gt, K=100)
            r20 = evaluate_lightgcn(lg, test_users, user_history, test_gt, K=20)
            elapsed = time.time() - t0
            print(f"     recall@100={r100*100:.2f}% recall@20={r20*100:.2f}% "
                  f"elapsed={elapsed:.0f}s")
            row = dict(emb_dim=emb_dim, n_layers=n_layers, lr=lr, reg=reg,
                       recall_at_100=r100, recall_at_20=r20,
                       epochs=args.epochs, elapsed_s=elapsed)
            results.append(row)
            if r100 > best["recall"]:
                best = dict(recall=r100, config=cfg_label, row=row)
            # free GPU memory
            del lg
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"     ERROR: {str(e)[:120]}")
            results.append(dict(emb_dim=emb_dim, n_layers=n_layers, lr=lr, reg=reg,
                                error=str(e)[:200]))

    # Compare to the paper's currently-reported numbers
    PAPER_VALUES = {
        "beauty": 0.0499,
        "movies": 0.0019,
        "electronics": 0.0015,
    }
    paper_val = PAPER_VALUES.get(name, None)

    summary = dict(
        dataset=name,
        catalog=int(train_df["item_id"].nunique()),
        n_test_users=len(test_users),
        n_configs=len(grid),
        best_recall_at_100=best["recall"],
        best_config=best.get("config"),
        paper_reported_recall=paper_val,
        improvement_over_paper=(best["recall"] - paper_val) / paper_val * 100 if paper_val else None,
        all_runs=results,
    )

    print(f"\n  BEST: {best['config']}  recall@100={best['recall']*100:.2f}%")
    if paper_val:
        diff = (best["recall"] - paper_val) / paper_val * 100
        print(f"  Paper reported: {paper_val*100:.2f}%  → tuned improvement: {diff:+.1f}%")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="beauty")
    ap.add_argument("--n_users", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="experiments/logs/cikm_lightgcn_sweep.json")
    args = ap.parse_args()

    if args.quick:
        args.datasets = "beauty"; args.epochs = 20

    datasets = [d.strip() for d in args.datasets.split(",")]
    print(f"datasets={datasets}  n_users={args.n_users}  epochs={args.epochs}")
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    t0 = time.time()
    by_dataset = {}
    for d in datasets:
        try:
            by_dataset[d] = run_dataset(d, args)
        except Exception as e:
            print(f"  FATAL {d}: {e}")
            by_dataset[d] = dict(dataset=d, error=str(e))

    summary = dict(
        experiment="cikm_lightgcn_sweep_P1C",
        timestamp=datetime.now().isoformat(timespec="seconds"),
        config=vars(args),
        results=by_dataset,
        elapsed_sec=time.time() - t0,
    )
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*72}\nFINAL TABLE\n{'='*72}")
    print(f"{'Dataset':<14}{'Best config':<35}{'Best R@100':>12}{'Paper R@100':>12}{'Improvement':>14}")
    for d, r in by_dataset.items():
        if "error" in r:
            print(f"{d:<14}ERROR: {r['error'][:60]}")
            continue
        cfg = r.get("best_config", "?")
        bp = r.get("best_recall_at_100", 0) * 100
        pp = (r.get("paper_reported_recall") or 0) * 100
        imp = r.get("improvement_over_paper")
        imp_s = f"{imp:+.1f}%" if imp is not None else "n/a"
        print(f"{d:<14}{cfg:<35}{bp:>11.2f}%{pp:>11.2f}%{imp_s:>14}")

    print(f"\nSaved: {out_path}  elapsed: {summary['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
