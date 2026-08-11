#!/usr/bin/env python3
"""
Rebuttal P0-C: Multi-model comparison at n=500 using LOCAL Ollama models only.
Replaces kdd_multi_model_unified.json's n=200 Beauty experiment with n=500.

Models: gemma3:4b, qwen3:8b (available locally via Ollama)
Dataset: Beauty (primary) + Movies (if time permits)

Usage: python scripts/rebuttal_n500_local.py
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json, re, time, requests
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.stats import wilcoxon

OLLAMA_URL  = "http://localhost:11434/api/chat"
LOCAL_MODELS = ["gemma3:4b", "qwen3:8b"]
N_USERS      = 500
K            = 100
TOP_K        = 10
SEED         = 42

DATASETS = {
    'beauty':  'data/processed/amazon_beauty_sampled',
    'movies':  'data/processed/amazon_movies_sampled',
}

PROMPT = """Based on the user's purchase history, rank these candidate items by relevance.

User's recent purchases:
{history}

Candidate items to rank:
{candidates}

Output the top {top_k} most relevant item numbers in order (most relevant first).
Format: Item <number> on each line."""


# ── CF retrieval ──────────────────────────────────────────────────────────────

class CFRetrieval:
    def fit(self, train_df):
        items = train_df['item_id'].unique()
        users = train_df['user_id'].unique()
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        self.user_to_idx = {u: idx for idx, u in enumerate(users)}
        rows = [self.user_to_idx[u] for u in train_df['user_id']]
        cols = [self.item_to_idx[i] for i in train_df['item_id']]
        mat  = csr_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(len(users), len(items)))
        n_comp = min(128, min(mat.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        self.user_factors = svd.fit_transform(mat)
        self.item_factors = svd.components_.T
        self.user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    def get_candidates(self, uid, K=100):
        if uid not in self.user_to_idx:
            return [], []
        uf = self.user_factors[self.user_to_idx[uid]]
        scores = np.dot(uf, self.item_factors.T)
        for item in self.user_history.get(uid, []):
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf
        top = np.argsort(scores)[::-1][:K]
        return [self.idx_to_item[i] for i in top], [float(scores[i]) for i in top]


# ── Metrics ───────────────────────────────────────────────────────────────────

def ndcg(ranked, gt, k=10):
    dcg  = sum(1/np.log2(i+2) for i,x in enumerate(ranked[:k]) if x in gt)
    idcg = sum(1/np.log2(i+2) for i in range(min(k,len(gt))))
    return dcg/idcg if idcg else 0.

def recall_at(cands, gt, k=100):
    return len(set(cands[:k]) & set(gt)) / len(gt) if gt else 0.

def bootstrap_ci(arr, n=1000):
    a = np.array(arr)
    b = [np.mean(np.random.choice(a, len(a))) for _ in range(n)]
    return float(np.mean(a)), float(np.percentile(b,2.5)), float(np.percentile(b,97.5))


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_ollama(model, prompt, timeout=60):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": [{"role":"user","content":prompt}],
            "stream": False,
            "options": {"temperature":0.1,"num_predict":300}
        }, timeout=timeout)
        return r.json()["message"]["content"]
    except Exception:
        return ""

def parse_ranking(resp, n):
    nums, seen, out = re.findall(r'\d+', resp), set(), []
    for x in nums:
        i = int(x) - 1
        if 0 <= i < n and i not in seen:
            out.append(i); seen.add(i)
    for i in range(n):
        if i not in seen: out.append(i)
    return out[:n]

def rerank(model, history_texts, cands_with_text, top_k=10):
    hist  = "\n".join(f"{i+1}. {t}" for i,t in enumerate(history_texts[-10:]))
    clist = "\n".join(f"{i+1}. {t[:60]}" for i,(c,t) in enumerate(cands_with_text[:30]))
    prompt = PROMPT.format(history=hist, candidates=clist, top_k=top_k)
    resp   = call_ollama(model, prompt)
    order  = parse_ranking(resp, len(cands_with_text))
    return [cands_with_text[i][0] for i in order]


# ── Main ──────────────────────────────────────────────────────────────────────

def run_dataset(ds_name, data_path):
    print(f"\n{'='*55}\n  {ds_name}\n{'='*55}")
    train = pd.read_parquet(project_root / data_path / 'train.parquet')
    test  = pd.read_parquet(project_root / data_path / 'test.parquet')

    # item texts
    item_texts = {}
    meta = project_root / data_path / 'item_metadata.json'
    if meta.exists():
        with open(meta) as f:
            for k,v in json.load(f).items():
                item_texts[int(k)] = str(v.get('text') or v.get('title',''))[:80]
    elif 'title' in train.columns:
        for _, row in train.drop_duplicates('item_id').iterrows():
            item_texts[row['item_id']] = str(row.get('title',''))[:80]

    cf = CFRetrieval(); cf.fit(train)

    np.random.seed(SEED)
    test_map = test.groupby('user_id')['item_id'].apply(list).to_dict()
    users = [u for u in test_map if u in cf.user_to_idx and test_map[u]]
    if len(users) > N_USERS:
        users = list(np.random.choice(users, N_USERS, replace=False))
    print(f"  Users: {len(users)}  Items: {train['item_id'].nunique()}")

    results = {}

    # CF baseline
    cf_ndcg, cf_recall = [], []
    for u in users:
        gt = set(test_map[u])
        cands, _ = cf.get_candidates(u, K)
        cf_ndcg.append(ndcg(cands, gt, TOP_K))
        cf_recall.append(recall_at(cands, gt, K))
    cf_m, cf_lo, cf_hi = bootstrap_ci(cf_ndcg)
    rec_m, rec_lo, rec_hi = bootstrap_ci(cf_recall)
    results['CF_Score'] = {
        'ndcg_mean': cf_m, 'ci': [cf_lo, cf_hi], 'n_samples': len(users),
        'recall_mean': rec_m, 'recall_ci': [rec_lo, rec_hi]
    }
    print(f"  CF Score: NDCG={cf_m:.4f}  Recall={rec_m:.4f}")

    # Each local model
    for model in LOCAL_MODELS:
        print(f"  Running {model} on {len(users)} users...")
        llm_ndcg = []
        for u in tqdm(users, desc=f"  {model}", leave=False):
            gt     = set(test_map[u])
            cands, scores = cf.get_candidates(u, K)
            hist   = [item_texts.get(x, f"Item {x}") for x in cf.user_history.get(u, [])[-10:]]
            cands_txt = [(c, item_texts.get(c, f"Item {c}")) for c in cands]
            ranked = rerank(model, hist, cands_txt, TOP_K)
            llm_ndcg.append(ndcg(ranked, gt, TOP_K))
        llm_m, llm_lo, llm_hi = bootstrap_ci(llm_ndcg)
        # Wilcoxon vs CF
        try:
            _, w_p = wilcoxon(cf_ndcg, llm_ndcg, zero_method='wilcox')
        except Exception:
            w_p = 1.0
        d = (llm_m - cf_m) / (np.std(llm_ndcg) + 1e-9)
        results[model.replace(':','_')] = {
            'ndcg_mean': llm_m, 'ci': [llm_lo, llm_hi],
            'vs_cf_pct': (llm_m - cf_m) / (cf_m + 1e-9) * 100,
            'cohens_d': float(d), 'wilcoxon_p': float(w_p),
            'n_samples': len(users)
        }
        print(f"    {model}: NDCG={llm_m:.4f}  vs CF={llm_m-cf_m:+.4f}  d={d:.3f}  p={w_p:.3f}")

    return results


def main():
    np.random.seed(SEED)
    all_results = {}
    for ds, path in DATASETS.items():
        if not (project_root / path).exists():
            print(f"[SKIP] {ds}: no data"); continue
        all_results[ds] = run_dataset(ds, path)

    out = project_root / 'experiments/logs/rebuttal_n500_local.json'
    with open(out, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(),
                   'config': {'n_users': N_USERS, 'K': K, 'top_k': TOP_K,
                               'models': LOCAL_MODELS, 'seed': SEED},
                   'results': all_results}, f, indent=2)
    print(f"\nSaved → {out}")

    # Print summary table
    print("\n--- Summary (Beauty) ---")
    print(f"{'Model':<25} {'NDCG':>8} {'vs CF':>8} {'d':>7} {'p':>7}")
    print("-"*55)
    for ds, res in all_results.items():
        if ds == 'beauty':
            cf_m = res['CF_Score']['ndcg_mean']
            for model, r in res.items():
                pct = r.get('vs_cf_pct', 0.)
                d   = r.get('cohens_d', 0.)
                p   = r.get('wilcoxon_p', 1.)
                print(f"{model:<25} {r['ndcg_mean']:>8.4f} {pct:>+7.1f}% {d:>7.3f} {p:>7.3f}")


if __name__ == '__main__':
    main()
