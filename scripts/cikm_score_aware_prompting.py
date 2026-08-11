#!/usr/bin/env python3
"""
CIKM 2026 camera-ready: score-aware prompting + LLM/CF ensemble.

Addresses the meta-reviewer's and Reviewer 3's request:
  "the LLM rerankers may be handicapped by not being shown CF scores or ranks,
   so a score-aware or LLM+CF ensemble baseline would harden the
   'LLMs overwrite the CF prior' claim"

Two prompt conditions, identical in every other respect:

  plain       - candidates as bare numbered titles. Reproduces the protocol used
                for the submitted paper's Table 5/7 numbers.
  scoreaware  - each candidate additionally carries the three structured ranking
                signals the reviewer named: its CF rank, its min-max-normalised
                CF score within the shown window, and its popularity percentile.

Control variables match the paper's main protocol exactly:
  SEED=42, K=100 retrieval, WINDOW=30 candidates shown to the LLM,
  TOP_K=10, SVD 128 factors, leave-one-out evaluation.

NOTE ON THE WINDOW. The paper reports K=100 as the retrieval budget. The LLM is
shown only the top-WINDOW=30 candidates (a context-length constraint inherited
from the submitted experiments); positions 31..100 are appended below the LLM's
output in their original CF order. Retrieval recall is therefore measured at 100,
but the ceiling actually binding the LLM's NDCG@10 is Recall@WINDOW. This script
reports eta against BOTH denominators so the camera-ready can state the correct
one.

Ensembles are computed offline from the stored per-user LLM permutation, so they
cost no additional API calls:
  rrf     - reciprocal rank fusion of the CF order and the LLM order (k=60)
  convex  - lambda * cf_score_norm + (1 - lambda) * llm_score_norm over a grid

IMPORTANT: lambda is NOT tuned on the evaluation set for the headline number.
We report the a-priori lambda=0.5 blend as the result, and additionally report
the best-lambda blend explicitly labelled as an oracle-tuned upper bound. This
paper retracted a previous finding caused by train/eval overlap; selecting a
hyper-parameter on the evaluation users and reporting it as a clean result would
be the same class of error.

Usage:
  python scripts/cikm_score_aware_prompting.py --test_n 20          # smoke test
  python scripts/cikm_score_aware_prompting.py                      # full n=500
  python scripts/cikm_score_aware_prompting.py --datasets beauty    # single ds
  python scripts/cikm_score_aware_prompting.py --offline_only       # re-fuse only

Output: experiments/logs/cikm_score_aware_prompting.json
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau, wilcoxon
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

try:
    import openai
except ImportError:
    print("pip install openai"); sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).parent.parent
CKPT_DIR = PROJECT / "experiments/logs/checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = PROJECT / "experiments/logs/cikm_score_aware_prompting.json"

# ── Config (must match the paper's main protocol) ─────────────────────────────
SEED = 42
K = 100          # retrieval candidate budget
WINDOW = 30      # candidates actually shown to the LLM
TOP_K = 10
N_FACTORS = 128
LAMBDA_GRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
LAMBDA_APRIORI = 0.5
RRF_K = 60

DATASETS = {
    "beauty":      "data/processed/amazon_beauty_sampled",
    "movies":      "data/processed/amazon_movies_sampled",
    "electronics": "data/processed/amazon_electronics_sampled",
}

MODEL = dict(
    provider="openai", key_env="OPENAI_API_KEY",
    base_url="https://api.openai.com/v1", model_id="gpt-4o-mini",
    price_in=0.15, price_out=0.60, delay=0.5,
)

PROMPT_PLAIN = (
    "Based on the user's purchase history, rank these candidate items by relevance.\n\n"
    "User's recent purchases:\n{history}\n\n"
    "Candidate items to rank:\n{candidates}\n\n"
    "Output only the top {top_k} most relevant item numbers in order (most relevant first). "
    "No explanation.\n"
    "Format: Item <number> on each line."
)

PROMPT_SCOREAWARE = (
    "Based on the user's purchase history, rank these candidate items by relevance.\n\n"
    "User's recent purchases:\n{history}\n\n"
    "Candidate items to rank. Each candidate is annotated with signals from the "
    "upstream collaborative-filtering retriever that produced this list:\n"
    "  cf_rank  - the retriever's own ranking of this candidate (1 = most relevant)\n"
    "  cf_score - the retriever's relevance score, rescaled to [0,1] within this list\n"
    "  pop_pct  - how popular the item is in the catalog (percentile, 99 = most popular)\n"
    "You may use, override, or ignore these signals as you see fit.\n\n"
    "{candidates}\n\n"
    "Output only the top {top_k} most relevant item numbers in order (most relevant first). "
    "No explanation.\n"
    "Format: Item <number> on each line."
)

CONDITIONS = {"plain": PROMPT_PLAIN, "scoreaware": PROMPT_SCOREAWARE}


# ── API helpers ───────────────────────────────────────────────────────────────

def _make_client():
    key = os.environ.get(MODEL["key_env"], "")
    if not key:
        env_path = PROJECT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith(MODEL["key_env"] + "="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise ValueError(f"API key {MODEL['key_env']} not found in environment or .env")
    return openai.OpenAI(api_key=key, base_url=MODEL["base_url"])


def call_llm(client, prompt, max_tokens=120, temperature=0.1, max_retries=5):
    """Returns (response_text, n_input_tokens, n_output_tokens)."""
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=MODEL["model_id"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature, timeout=60,
            )
            text = r.choices[0].message.content or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text, r.usage.prompt_tokens, r.usage.completion_tokens
        except openai.RateLimitError:
            wait = 2 ** attempt + np.random.uniform(0, 1)
            print(f"\n  [RateLimit] attempt {attempt+1}/{max_retries}, waiting {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"\n  [FAIL] {str(e)[:120]}")
                return "", 0, 0
    return "", 0, 0


def parse_ranking(response: str, n_cands: int) -> list[int]:
    """Parse item numbers from the LLM response into 0-based indices.

    Identical to the submitted paper's parser: indices the LLM did not name are
    appended afterwards in their original (CF) order.
    """
    nums, seen, out = re.findall(r"\d+", response), set(), []
    for x in nums:
        i = int(x) - 1
        if 0 <= i < n_cands and i not in seen:
            out.append(i)
            seen.add(i)
    for i in range(n_cands):
        if i not in seen:
            out.append(i)
    return out[:n_cands]


# ── Metrics ───────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_ids, gt_ids, k=TOP_K):
    dcg = sum(1 / np.log2(i + 2) for i, x in enumerate(ranked_ids[:k]) if x in gt_ids)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(gt_ids))))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(ranked_ids, gt_ids, k=TOP_K):
    return 1.0 if set(ranked_ids[:k]) & set(gt_ids) else 0.0


def recall_at_k(cands, gt, k):
    return len(set(cands[:k]) & set(gt)) / len(gt) if gt else 0.0


def bootstrap_ci(arr, n=2000, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(arr, dtype=float)
    samples = rng.choice(a, (n, len(a)), replace=True).mean(axis=1)
    return (float(a.mean()),
            float(np.percentile(samples, 100 * alpha / 2)),
            float(np.percentile(samples, 100 * (1 - alpha / 2))))


def safe_wilcoxon(a, b):
    try:
        if np.allclose(a, b):
            return float("nan")
        return float(wilcoxon(a, b, zero_method="wilcox")[1])
    except Exception:
        return float("nan")


# ── CF retrieval (identical to the paper's CF-SVD retriever) ──────────────────

class CFRetrieval:
    def fit(self, train_df):
        items = train_df["item_id"].unique()
        users = train_df["user_id"].unique()
        self.item_to_idx = {v: i for i, v in enumerate(items)}
        self.idx_to_item = {i: v for v, i in self.item_to_idx.items()}
        self.user_to_idx = {v: i for i, v in enumerate(users)}
        rows = [self.user_to_idx[u] for u in train_df["user_id"]]
        cols = [self.item_to_idx[i] for i in train_df["item_id"]]
        mat = csr_matrix((np.ones(len(rows)), (rows, cols)),
                         shape=(len(users), len(items)))
        n_comp = min(N_FACTORS, min(mat.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        self.U = svd.fit_transform(mat)
        self.V = svd.components_.T
        self.history = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
        # Popularity percentile per item, used as a structured signal in the prompt
        counts = train_df["item_id"].value_counts()
        ranks = counts.rank(pct=True)
        self.pop_pct = {int(i): int(round(float(p) * 99)) for i, p in ranks.items()}

    def get_candidates(self, uid, k=K):
        if uid not in self.user_to_idx:
            return [], []
        uf = self.U[self.user_to_idx[uid]]
        scores = self.V @ uf
        for item in self.history.get(uid, []):
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[i] for i in top_idx], [float(scores[i]) for i in top_idx]


# ── Ensemble fusion (offline, no API cost) ────────────────────────────────────

def fuse_rrf(llm_order, n, k=RRF_K):
    """Reciprocal rank fusion of the CF order (identity) and the LLM order."""
    rank_llm = {item: pos for pos, item in enumerate(llm_order)}
    scores = {i: 1.0 / (k + i + 1) + 1.0 / (k + rank_llm.get(i, n - 1) + 1)
              for i in range(n)}
    return sorted(range(n), key=lambda i: -scores[i])


def fuse_convex(llm_order, n, lam):
    """lam * CF positional score + (1 - lam) * LLM positional score.

    lam = 1.0 recovers pure CF, lam = 0.0 recovers pure LLM.
    """
    if n <= 1:
        return list(range(n))
    rank_llm = {item: pos for pos, item in enumerate(llm_order)}
    scores = {}
    for i in range(n):
        cf_s = 1.0 - i / (n - 1)
        llm_s = 1.0 - rank_llm.get(i, n - 1) / (n - 1)
        scores[i] = lam * cf_s + (1 - lam) * llm_s
    # Ties broken by CF order, which is the conservative choice
    return sorted(range(n), key=lambda i: (-scores[i], i))


# ── Checkpointing ─────────────────────────────────────────────────────────────

def ckpt_path(condition, dataset):
    return CKPT_DIR / f"scoreaware_{condition}_{dataset}.jsonl"


def load_checkpoint(condition, dataset):
    p = ckpt_path(condition, dataset)
    done = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["user_id"]] = rec
    return done


def append_checkpoint(condition, dataset, record):
    with open(ckpt_path(condition, dataset), "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Prompt rendering ──────────────────────────────────────────────────────────

def render_candidates(condition, window_items, window_scores, pop_pct, item_texts):
    """Render the candidate block for the given condition."""
    if condition == "plain":
        return "\n".join(
            f"{i+1}. {item_texts.get(c, f'Item {c}')[:60]}"
            for i, c in enumerate(window_items))

    # scoreaware: min-max normalise the CF score within the shown window
    s = np.asarray(window_scores, dtype=float)
    finite = s[np.isfinite(s)]
    if finite.size and finite.max() > finite.min():
        norm = (s - finite.min()) / (finite.max() - finite.min())
    else:
        norm = np.ones_like(s)
    norm = np.clip(np.nan_to_num(norm, nan=0.0), 0.0, 1.0)
    return "\n".join(
        f"{i+1}. {item_texts.get(c, f'Item {c}')[:60]} "
        f"(cf_rank={i+1}, cf_score={norm[i]:.2f}, pop_pct={pop_pct.get(int(c), 0)})"
        for i, c in enumerate(window_items))


# ── Per-dataset run ───────────────────────────────────────────────────────────

def load_dataset(data_path):
    data_dir = PROJECT / data_path
    train = pd.read_parquet(data_dir / "train.parquet")
    test_file = (data_dir / "test_loo.parquet"
                 if (data_dir / "test_loo.parquet").exists()
                 else data_dir / "test.parquet")
    test = pd.read_parquet(test_file)

    item_texts = {}
    meta = data_dir / "item_metadata.json"
    if meta.exists():
        for k, v in json.loads(meta.read_text()).items():
            if isinstance(v, dict):
                title = v.get("text") or v.get("title", "")
                genres = v.get("genres", "")
                text = f"{title} [{genres}]" if genres else title
            else:
                text = str(v)
            item_texts[int(k)] = str(text)[:80]
    elif "title" in train.columns:
        for _, row in train.drop_duplicates("item_id").iterrows():
            item_texts[row["item_id"]] = str(row.get("title", ""))[:80]
    return train, test, item_texts


def run_dataset(ds_name, data_path, test_n, conditions, offline_only):
    print(f"\n{'='*66}\n  Dataset: {ds_name}\n{'='*66}")
    train, test, item_texts = load_dataset(data_path)
    cf = CFRetrieval()
    cf.fit(train)

    # User sample: identical draw to the paper's canonical n=500 sample
    np.random.seed(SEED)
    test_map = test.groupby("user_id")["item_id"].apply(list).to_dict()
    users = [u for u in test_map if u in cf.user_to_idx and test_map[u]]
    if test_n:
        users = users[:test_n]
    elif len(users) > 500:
        users = list(np.random.choice(users, 500, replace=False))
    print(f"  Users: {len(users)}  Items: {train['item_id'].nunique()}")

    # Precompute CF candidates, baseline metrics, and both recall denominators
    cands_by_user, scores_by_user = {}, {}
    cf_ndcg, cf_hit, rec_K, rec_W = [], [], [], []
    for u in users:
        gt = set(test_map[u])
        cands, scores = cf.get_candidates(u, K)
        cands_by_user[u], scores_by_user[u] = cands, scores
        cf_ndcg.append(ndcg_at_k(cands, gt))
        cf_hit.append(hit_at_k(cands, gt))
        rec_K.append(recall_at_k(cands, gt, K))
        rec_W.append(recall_at_k(cands, gt, WINDOW))

    cf_m, cf_lo, cf_hi = bootstrap_ci(cf_ndcg)
    recK_m = float(np.mean(rec_K))
    recW_m = float(np.mean(rec_W))
    print(f"  CF-SVD   NDCG@10={cf_m:.4f} [{cf_lo:.4f},{cf_hi:.4f}]")
    print(f"  Recall@{K}={recK_m*100:.2f}%   Recall@{WINDOW}={recW_m*100:.2f}%  "
          f"(the ceiling actually binding the LLM)")

    ds_out = {
        "n_users": len(users),
        "recall_at_K": recK_m, "recall_at_window": recW_m,
        "K": K, "window": WINDOW,
        "cf": {"ndcg_mean": cf_m, "ci": [cf_lo, cf_hi],
               "hit_mean": float(np.mean(cf_hit)),
               "eta_vs_recallK": cf_m / recK_m if recK_m > 0 else None,
               "eta_vs_recallW": cf_m / recW_m if recW_m > 0 else None},
        "conditions": {},
    }

    client = None
    if not offline_only:
        client = _make_client()

    for cond in conditions:
        done = load_checkpoint(cond, ds_name)
        todo = [u for u in users if str(u) not in done]
        if offline_only and todo:
            print(f"\n  [{cond}] SKIP: offline_only but {len(todo)} users missing")
            continue
        print(f"\n  [{cond}] {len(done)} cached, {len(todo)} to run")

        tok_in = tok_out = 0
        for u in tqdm(todo, desc=f"  {cond}", leave=False):
            cands = cands_by_user[u]
            scores = scores_by_user[u]
            win_items = cands[:WINDOW]
            win_scores = scores[:WINDOW]
            hist = [item_texts.get(x, f"Item {x}")
                    for x in cf.history.get(u, [])[-10:]]
            history_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(hist))
            candidates_str = render_candidates(cond, win_items, win_scores,
                                               cf.pop_pct, item_texts)
            prompt = CONDITIONS[cond].format(
                history=history_str, candidates=candidates_str, top_k=TOP_K)

            t0 = time.time()
            resp, n_in, n_out = call_llm(client, prompt)
            latency = time.time() - t0
            if latency < MODEL["delay"]:
                time.sleep(MODEL["delay"] - latency)

            order = parse_ranking(resp, len(win_items))
            tok_in += n_in
            tok_out += n_out
            append_checkpoint(cond, ds_name, {
                "user_id": str(u), "llm_order": order,
                "n_in": n_in, "n_out": n_out, "latency_s": round(latency, 2),
            })

        # ── Score this condition and its offline fusions ──────────────────────
        ckpt = load_checkpoint(cond, ds_name)
        missing = [u for u in users if str(u) not in ckpt]
        if missing:
            print(f"  [{cond}] INCOMPLETE: {len(missing)} users missing, skipping")
            continue

        llm_ndcg, llm_hit, taus = [], [], []
        rrf_ndcg = []
        convex_ndcg = {lam: [] for lam in LAMBDA_GRID}

        for u in users:
            gt = set(test_map[u])
            cands = cands_by_user[u]
            win_items = cands[:WINDOW]
            tail = cands[WINDOW:]
            n = len(win_items)
            order = ckpt[str(u)]["llm_order"][:n]
            # Guard against a truncated/duplicated permutation
            if sorted(order) != list(range(n)):
                seen, fixed = set(order), list(order)
                fixed += [i for i in range(n) if i not in seen]
                order = fixed[:n]

            ranked = [win_items[i] for i in order] + tail
            llm_ndcg.append(ndcg_at_k(ranked, gt))
            llm_hit.append(hit_at_k(ranked, gt))
            if n > 1:
                t = kendalltau(list(range(n)), order)[0]
                if not np.isnan(t):
                    taus.append(float(t))

            rrf_order = fuse_rrf(order, n)
            rrf_ndcg.append(ndcg_at_k([win_items[i] for i in rrf_order] + tail, gt))
            for lam in LAMBDA_GRID:
                cx = fuse_convex(order, n, lam)
                convex_ndcg[lam].append(
                    ndcg_at_k([win_items[i] for i in cx] + tail, gt))

        m, lo, hi = bootstrap_ci(llm_ndcg)
        rm, rlo, rhi = bootstrap_ci(rrf_ndcg)
        ap = convex_ndcg[LAMBDA_APRIORI]
        am, alo, ahi = bootstrap_ci(ap)
        best_lam = max(LAMBDA_GRID, key=lambda l: float(np.mean(convex_ndcg[l])))
        cost = (tok_in * MODEL["price_in"] + tok_out * MODEL["price_out"]) / 1e6

        ds_out["conditions"][cond] = {
            "llm": {
                "ndcg_mean": m, "ci": [lo, hi],
                "hit_mean": float(np.mean(llm_hit)),
                "vs_cf_pct": (m - cf_m) / (cf_m + 1e-12) * 100,
                "wilcoxon_p": safe_wilcoxon(cf_ndcg, llm_ndcg),
                "mean_kendall_tau": float(np.mean(taus)) if taus else None,
                "eta_vs_recallK": m / recK_m if recK_m > 0 else None,
                "eta_vs_recallW": m / recW_m if recW_m > 0 else None,
            },
            "ensemble_rrf": {
                "ndcg_mean": rm, "ci": [rlo, rhi],
                "vs_cf_pct": (rm - cf_m) / (cf_m + 1e-12) * 100,
                "wilcoxon_p": safe_wilcoxon(cf_ndcg, rrf_ndcg),
            },
            "ensemble_convex_apriori": {
                "lambda": LAMBDA_APRIORI,
                "ndcg_mean": am, "ci": [alo, ahi],
                "vs_cf_pct": (am - cf_m) / (cf_m + 1e-12) * 100,
                "wilcoxon_p": safe_wilcoxon(cf_ndcg, ap),
            },
            "ensemble_convex_oracle_tuned": {
                "note": "lambda selected on the evaluation users; an upper bound, "
                        "NOT a clean held-out result",
                "lambda": best_lam,
                "ndcg_mean": float(np.mean(convex_ndcg[best_lam])),
            },
            "convex_lambda_curve": {
                str(l): float(np.mean(convex_ndcg[l])) for l in LAMBDA_GRID},
            "tokens": {"in": tok_in, "out": tok_out},
            "cost_usd": round(cost, 4),
        }

        tau_s = ds_out["conditions"][cond]["llm"]["mean_kendall_tau"]
        print(f"  [{cond}] LLM NDCG={m:.4f} ({(m-cf_m)/(cf_m+1e-12)*100:+.1f}% vs CF) "
              f"p={ds_out['conditions'][cond]['llm']['wilcoxon_p']:.3f} "
              f"tau={tau_s if tau_s is None else round(tau_s, 3)}")
        print(f"  [{cond}] RRF={rm:.4f}  convex(l=0.5)={am:.4f}  "
              f"best-lambda={best_lam} -> {np.mean(convex_ndcg[best_lam]):.4f}  "
              f"cost=${cost:.3f}")

    return ds_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_n", type=int, default=None,
                    help="Run on the first N users only (smoke test)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()))
    ap.add_argument("--offline_only", action="store_true",
                    help="Re-score and re-fuse from checkpoints, no API calls")
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    all_results = {}
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"[SKIP] unknown dataset {ds}")
            continue
        if not (PROJECT / DATASETS[ds]).exists():
            print(f"[SKIP] {ds}: data not found at {DATASETS[ds]}")
            continue
        all_results[ds] = run_dataset(ds, DATASETS[ds], args.test_n,
                                      args.conditions, args.offline_only)
        with open(args.out, "w") as f:
            json.dump({
                "experiment": "cikm_score_aware_prompting",
                "timestamp": datetime.now().isoformat(),
                "model": MODEL["model_id"],
                "config": {"seed": SEED, "K": K, "window": WINDOW,
                           "top_k": TOP_K, "n_factors": N_FACTORS,
                           "lambda_apriori": LAMBDA_APRIORI, "rrf_k": RRF_K,
                           "n_users": args.test_n or 500},
                "results": all_results,
            }, f, indent=2)
        print(f"\n  Saved -> {args.out}")

    print("\n" + "=" * 66)
    print("SUMMARY  (NDCG@10; eta_W = NDCG / Recall@30, the binding ceiling)")
    print("=" * 66)
    for ds, r in all_results.items():
        print(f"\n  {ds.upper()}  Recall@{K}={r['recall_at_K']*100:.2f}%  "
              f"Recall@{WINDOW}={r['recall_at_window']*100:.2f}%")
        print(f"    {'method':<28}{'NDCG':>9}{'vs CF':>9}{'p':>8}{'eta_W':>8}")
        cf = r["cf"]
        print(f"    {'CF-SVD':<28}{cf['ndcg_mean']:>9.4f}{'--':>9}{'--':>8}"
              f"{(cf['eta_vs_recallW'] or 0)*100:>7.1f}%")
        for cond, c in r["conditions"].items():
            for label, key in (("LLM", "llm"), ("+CF RRF", "ensemble_rrf"),
                               ("+CF convex(0.5)", "ensemble_convex_apriori")):
                e = c[key]
                eta = e.get("eta_vs_recallW")
                eta_s = f"{eta*100:>7.1f}%" if eta else f"{'':>8}"
                p = e.get("wilcoxon_p")
                p_s = f"{p:>8.3f}" if p is not None and not np.isnan(p) else f"{'n.s.':>8}"
                print(f"    {cond + ' ' + label:<28}{e['ndcg_mean']:>9.4f}"
                      f"{e['vs_cf_pct']:>+8.1f}%{p_s}{eta_s}")


if __name__ == "__main__":
    main()
