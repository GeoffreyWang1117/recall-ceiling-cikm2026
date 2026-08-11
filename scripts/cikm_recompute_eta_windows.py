#!/usr/bin/env python3
"""
CIKM 2026 camera-ready: recompute the ceiling-utilisation ratio eta against each
reranker's own reranking window.

The submitted paper divided every reranker's NDCG@10 by Recall@100. That is the
correct denominator only for rerankers that actually see all 100 candidates.
The listwise LLM prompts show the top-30 CF candidates and append positions
31..100 below the LLM's output in CF order, so an LLM's NDCG@10 is bounded by
Recall@30, not Recall@100 (Theorem 1 / Definition 1 in the paper).

Window per method, read off the experiment code:
  30   zero-shot listwise LLM prompts
         scripts/rebuttal_multiapi_n500.py:364  (cands_txt[:30])
         scripts/aggregate_v4_results.py:181    (n_shown = min(30, len(cands)))
  100  pointwise / feature-based rerankers, which score every candidate
         scripts/cikm_neural_rerankers_disjoint.py:290  (argsort over all cands)
         scripts/cikm_finetuned_llm_reranker.py:229-231 (one prompt per cand)
  100  CF-SVD itself (the retrieval order)

Usage:
  python scripts/cikm_recompute_eta_windows.py
Output:
  experiments/logs/cikm_eta_windows.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

PROJECT = Path(__file__).parent.parent
OUT = PROJECT / "experiments/logs/cikm_eta_windows.json"

SEED, K, WINDOW, N_FACTORS, TOP_K = 42, 100, 30, 128, 10
DATA = "data/processed/amazon_beauty_sampled"

# (display name, source json, json path to the NDCG@10 value, window)
SOURCES = [
    ("CF-SVD",        "rebuttal_multiapi_n500.json", ("beauty", "CF_Score"), 100),
    ("Gemma3-4B",     "rebuttal_n500_local.json",    ("beauty", "gemma3_4b"), 30),
    ("Qwen3-8B",      "rebuttal_n500_local.json",    ("beauty", "qwen3_8b"), 30),
    ("GPT-4o-mini",   "rebuttal_multiapi_n500.json", ("beauty", "gpt-4o-mini"), 30),
    ("Qwen3-32B",     "rebuttal_multiapi_n500.json", ("beauty", "qwen3-32b"), 30),
    ("Llama-3.3-70B", "rebuttal_multiapi_n500.json", ("beauty", "llama-3.3-70b-versatile"), 30),
    ("Kimi-K2",       "rebuttal_multiapi_n500.json", ("beauty", "kimi-k2-turbo-preview"), 30),
    ("DeepSeek-V3",   "rebuttal_multiapi_n500.json", ("beauty", "deepseek-chat"), 30),
]
NEURAL_SRC = "cikm_neural_rerankers_disjoint.json"
LORA_SRC = "cikm_finetune_llm_n500.json"


def load_json(name):
    """Prefer the committed version; later runs overwrote some result files."""
    import subprocess
    rel = f"experiments/logs/{name}"
    try:
        txt = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=PROJECT,
                             capture_output=True, text=True, check=True).stdout
        return json.loads(txt)
    except Exception:
        return json.loads((PROJECT / rel).read_text())


def recall_at(cands, gt, k):
    return len(set(cands[:k]) & set(gt)) / len(gt) if gt else 0.0


def compute_recalls():
    d = PROJECT / DATA
    train = pd.read_parquet(d / "train.parquet")
    test = pd.read_parquet(d / "test.parquet")

    items = train["item_id"].unique()
    users_all = train["user_id"].unique()
    i2x = {v: i for i, v in enumerate(items)}
    u2x = {v: i for i, v in enumerate(users_all)}
    x2i = {i: v for v, i in i2x.items()}
    rows = [u2x[u] for u in train["user_id"]]
    cols = [i2x[i] for i in train["item_id"]]
    mat = csr_matrix((np.ones(len(rows)), (rows, cols)),
                     shape=(len(users_all), len(items)))
    svd = TruncatedSVD(n_components=min(N_FACTORS, min(mat.shape) - 1),
                       random_state=SEED)
    U, V = svd.fit_transform(mat), svd.components_.T
    hist = train.groupby("user_id")["item_id"].apply(list).to_dict()

    np.random.seed(SEED)
    test_map = test.groupby("user_id")["item_id"].apply(list).to_dict()
    users = [u for u in test_map if u in u2x and test_map[u]]
    if len(users) > 500:
        users = list(np.random.choice(users, 500, replace=False))

    rK, rW = [], []
    for u in users:
        s = V @ U[u2x[u]]
        for it in hist.get(u, []):
            if it in i2x:
                s[i2x[it]] = -np.inf
        cands = [x2i[i] for i in np.argsort(s)[::-1][:K]]
        gt = set(test_map[u])
        rK.append(recall_at(cands, gt, K))
        rW.append(recall_at(cands, gt, WINDOW))
    return float(np.mean(rK)), float(np.mean(rW)), len(users)


def main():
    recall_K, recall_W, n = compute_recalls()
    denom = {100: recall_K, 30: recall_W}
    print(f"Beauty n={n}  Recall@{K}={recall_K*100:.2f}%  Recall@{WINDOW}={recall_W*100:.2f}%\n")

    rows = []
    for name, src, (ds, key), win in SOURCES:
        blob = load_json(src)
        entry = blob.get("results", blob).get(ds, {}).get(key)
        if not entry or "ndcg_mean" not in entry:
            print(f"  [skip] {name}: not found in {src}")
            continue
        rows.append((name, entry["ndcg_mean"], win))

    neural = load_json(NEURAL_SRC)
    res = neural.get("results", neural)
    beauty = next((x for x in (res if isinstance(res, list) else res.values())
                   if isinstance(x, dict) and x.get("dataset") == "beauty"), None)
    if beauty:
        for rn, rv in beauty.get("rerankers", {}).items():
            if "ndcg" in rv:
                rows.append((rn, rv["ndcg"], 100))

    lora = load_json(LORA_SRC)["results"]["amazon_beauty_sampled"]
    rows.append(("LoRA-LLaMA-3B", lora["ft_ndcg"], 100))

    print(f"{'method':<16}{'NDCG@10':>9}{'|W|':>6}{'eta_old':>9}{'eta_new':>9}")
    out = []
    for name, ndcg, win in rows:
        eta_old = ndcg / recall_K
        eta_new = ndcg / denom[win]
        out.append(dict(method=name, ndcg=ndcg, window=win,
                        eta_vs_recall100=eta_old, eta_vs_own_window=eta_new))
        print(f"{name:<16}{ndcg:>9.4f}{win:>6}{eta_old*100:>8.1f}%{eta_new*100:>8.1f}%")

    llm = [r for r in out if r["window"] == 30]
    full = [r for r in out if r["window"] == 100 and r["method"] != "CF-SVD"]
    summary = dict(
        llm_eta_min=min(r["eta_vs_own_window"] for r in llm),
        llm_eta_max=max(r["eta_vs_own_window"] for r in llm),
        fullwindow_eta_min=min(r["eta_vs_own_window"] for r in full),
        fullwindow_eta_max=max(r["eta_vs_own_window"] for r in full),
        overall_eta_max=max(r["eta_vs_own_window"] for r in out),
    )
    print(f"\nLLMs (|W|=30):        eta {summary['llm_eta_min']*100:.1f}%--"
          f"{summary['llm_eta_max']*100:.1f}%")
    print(f"Full-window rerankers: eta {summary['fullwindow_eta_min']*100:.1f}%--"
          f"{summary['fullwindow_eta_max']*100:.1f}%")
    print(f"Overall max:           eta {summary['overall_eta_max']*100:.1f}%")

    OUT.write_text(json.dumps(dict(
        experiment="cikm_eta_windows", dataset="beauty", n_users=n,
        recall_at_100=recall_K, recall_at_30=recall_W,
        methods=out, summary=summary), indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
