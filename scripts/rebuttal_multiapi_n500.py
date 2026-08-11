#!/usr/bin/env python3
"""
Rebuttal multi-model comparison at n=500 using cloud APIs.
Providers: OpenAI, Fireworks AI, Groq.

Control variables (identical to kdd_oracle_realistic_n500.py):
  SEED=42, K=100, TOP_K=10, SVD 128 factors, P1 basic prompt, leave-one-out eval.

Features:
  - Per-user checkpoint/resume: results saved to JSONL immediately after each user.
    If interrupted, re-run the same command to resume from where it stopped.
  - Incremental cost tracking per model.
  - Exponential backoff on rate limits (HTTP 429).
  - --test_n N to run small-scale pilot before full run.

Usage:
  # Dry run / small test (n=20)
  python scripts/rebuttal_multiapi_n500.py --test_n 20

  # Full run (n=500)
  python scripts/rebuttal_multiapi_n500.py

  # Resume after interruption (re-run same command):
  python scripts/rebuttal_multiapi_n500.py
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
from scipy.stats import wilcoxon
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

try:
    import openai
except ImportError:
    print("pip install openai"); sys.exit(1)

try:
    from scipy.stats import kendalltau as _kendalltau
    HAS_SCIPY_TAU = True
except ImportError:
    HAS_SCIPY_TAU = False

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).parent.parent
CKPT_DIR = PROJECT / "experiments/logs/checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = PROJECT / "experiments/logs/rebuttal_multiapi_n500.json"

# ── Config ────────────────────────────────────────────────────────────────────
SEED   = 42
K      = 100
TOP_K  = 10
N_FACTORS = 128

DATASETS = {
    "beauty":      "data/processed/amazon_beauty_sampled",
    "movies":      "data/processed/amazon_movies_sampled",
    "electronics": "data/processed/amazon_electronics_sampled",
    "mind":        "data/processed/mind_news",
    "movielens":   "data/processed/movielens_25m",
}

# Models: (provider_tag, api_key_env, base_url, model_id, $/M-in, $/M-out)
MODELS = [
    # (provider_tag, api_key_env, base_url, model_id, $/M-in, $/M-out)
    ("openai",    "OPENAI_API_KEY",   "https://api.openai.com/v1",
     "gpt-4o-mini",                                               0.15,  0.60),
    ("groq",      "GROQ_API_KEY",     "https://api.groq.com/openai/v1",
     "llama-3.3-70b-versatile",                                   0.59,  0.79),
    ("groq",      "GROQ_API_KEY",     "https://api.groq.com/openai/v1",
     "qwen/qwen3-32b",                                            0.29,  0.39),
    # Official DeepSeek API (deepseek-chat = DeepSeek-V3): 66% cheaper than Fireworks hosting
    ("deepseek",  "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1",
     "deepseek-chat",                                             0.27,  1.10),
    # DeepSeek V4 family — -pro at 75% launch discount until 2026-05-05 15:59 UTC.
    # 8-tuple: (..., ckpt_tag, extra_body, max_tokens). Two variants per model:
    # nothink = apples-to-apples vs other LLMs, think = real reasoning test.
    ("deepseek",  "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1",
     "deepseek-v4-pro",                                           0.435, 0.87,
     "deepseek-v4-pro-nothink",   {"thinking": {"type": "disabled"}}, 120),
    ("deepseek",  "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1",
     "deepseek-v4-pro",                                           0.435, 0.87,
     "deepseek-v4-pro-think",     {"thinking": {"type": "enabled"}},  32000),
    ("deepseek",  "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1",
     "deepseek-v4-flash",                                         0.14,  0.28,
     "deepseek-v4-flash-nothink", {"thinking": {"type": "disabled"}}, 120),
    ("deepseek",  "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1",
     "deepseek-v4-flash",                                         0.14,  0.28,
     "deepseek-v4-flash-think",   {"thinking": {"type": "enabled"}},  32000),
    # Official Moonshot/Kimi API: kimi-k2-turbo-preview, faster and cheaper than Fireworks
    ("kimi",      "KIMI_API_KEY",     "https://api.moonshot.ai/v1",
     "kimi-k2-turbo-preview",                                     0.50,  2.50),
]

# Minimum inter-request delay (seconds) per provider to avoid rate-limit 429s.
PROVIDER_DELAY = {
    "openai":   0.5,
    "groq":     2.0,   # ~30 req/min for 70B model
    "deepseek": 0.3,
    "kimi":     0.5,
}

PROMPT = (
    "Based on the user's purchase history, rank these candidate items by relevance.\n\n"
    "User's recent purchases:\n{history}\n\n"
    "Candidate items to rank:\n{candidates}\n\n"
    "Output only the top {top_k} most relevant item numbers in order (most relevant first). "
    "No explanation.\n"
    "Format: Item <number> on each line."
)

# ── API helpers ───────────────────────────────────────────────────────────────

def _make_client(provider_tag, base_url, key_env):
    key = os.environ.get(key_env, "")
    if not key:
        # Try loading from project .env
        env_path = PROJECT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith(key_env + "="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise ValueError(f"API key {key_env} not found in environment or .env")
    return openai.OpenAI(api_key=key, base_url=base_url)


def call_llm(client, model_id, prompt, max_tokens=120, temperature=0.1,
             max_retries=5, extra_body=None) -> tuple[str, int, int]:
    """Returns (response_text, n_input_tokens, n_output_tokens).
    Retries on 429 with exponential backoff."""
    # Scale per-call timeout with max_tokens; thinking models at 32K need ~4-5 min.
    per_call_timeout = max(60, int(max_tokens * 0.05) + 30)
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=per_call_timeout,
            )
            if extra_body:
                kwargs["extra_body"] = extra_body
            r = client.chat.completions.create(**kwargs)
            text = r.choices[0].message.content or ""
            # Strip <think>…</think> blocks (Qwen3, DeepSeek R1 style)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text, r.usage.prompt_tokens, r.usage.completion_tokens
        except openai.RateLimitError:
            wait = 2 ** attempt + np.random.uniform(0, 1)
            print(f"\n  [RateLimit] {model_id} attempt {attempt+1}/{max_retries}, "
                  f"waiting {wait:.1f}s…")
            time.sleep(wait)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"\n  [Error] {model_id}: {str(e)[:80]}, retry in {wait}s…")
                time.sleep(wait)
            else:
                print(f"\n  [FAIL] {model_id}: {str(e)[:120]}")
                return "", 0, 0
    return "", 0, 0


def parse_ranking(response: str, n_cands: int) -> list[int]:
    """Parse 'Item N' lines from LLM response into 0-based indices."""
    nums, seen, out = re.findall(r"\d+", response), set(), []
    for x in nums:
        i = int(x) - 1
        if 0 <= i < n_cands and i not in seen:
            out.append(i)
            seen.add(i)
    # Fill remaining positions in original order
    for i in range(n_cands):
        if i not in seen:
            out.append(i)
    return out[:n_cands]

# ── Metrics ───────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_ids, gt_ids, k=10):
    dcg  = sum(1 / np.log2(i + 2) for i, x in enumerate(ranked_ids[:k]) if x in gt_ids)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(gt_ids))))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(cands, gt, k=100):
    return len(set(cands[:k]) & set(gt)) / len(gt) if gt else 0.0


def bootstrap_ci(arr, n=2000, alpha=0.05):
    a = np.array(arr)
    samples = np.random.choice(a, (n, len(a)), replace=True).mean(axis=1)
    return float(a.mean()), float(np.percentile(samples, 100 * alpha / 2)), \
           float(np.percentile(samples, 100 * (1 - alpha / 2)))

# ── CF Retrieval ──────────────────────────────────────────────────────────────

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
        scores = self.V @ uf  # shape (n_items,) — note: V is (n_items, n_comp)
        # Actually U is (n_users, n_comp), V is (n_items, n_comp)
        # scores = U[uid] @ V.T — already done above via V @ uf since V = components_.T
        for item in self.history.get(uid, []):
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[i] for i in top_idx], [float(scores[i]) for i in top_idx]

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def ckpt_path(model_key: str, dataset: str) -> Path:
    safe = model_key.replace("/", "_").replace(":", "_")
    return CKPT_DIR / f"multiapi_{safe}_{dataset}.jsonl"


def load_checkpoint(model_key: str, dataset: str) -> dict[str, dict]:
    """Returns {user_id: result_dict} for already-completed users."""
    p = ckpt_path(model_key, dataset)
    done = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["user_id"]] = rec
    return done


def append_checkpoint(model_key: str, dataset: str, record: dict):
    """Append one user's result to the checkpoint file immediately."""
    p = ckpt_path(model_key, dataset)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")

# ── Main experiment ───────────────────────────────────────────────────────────

def run_dataset(ds_name: str, data_path: str, test_n: int | None,
                models_to_run: list) -> dict:
    print(f"\n{'='*60}\n  Dataset: {ds_name}\n{'='*60}")
    data_dir = PROJECT / data_path
    train = pd.read_parquet(data_dir / "train.parquet")
    # MovieLens uses test_loo.parquet for leave-one-out evaluation
    test_file = data_dir / "test_loo.parquet" if (data_dir / "test_loo.parquet").exists() \
                else data_dir / "test.parquet"
    test  = pd.read_parquet(test_file)

    # Item texts
    item_texts: dict[int, str] = {}
    meta = data_dir / "item_metadata.json"
    if meta.exists():
        for k, v in json.loads(meta.read_text()).items():
            if isinstance(v, dict):
                title  = v.get("text") or v.get("title", "")
                genres = v.get("genres", "")
                # For MovieLens: "Toy Story (1995) [Adventure|Animation|Comedy]"
                text = f"{title} [{genres}]" if genres else title
            else:
                text = str(v)
            item_texts[int(k)] = str(text)[:80]
    elif "title" in train.columns:
        for _, row in train.drop_duplicates("item_id").iterrows():
            item_texts[row["item_id"]] = str(row.get("title", ""))[:80]

    cf = CFRetrieval()
    cf.fit(train)

    np.random.seed(SEED)
    test_map = test.groupby("user_id")["item_id"].apply(list).to_dict()
    users = [u for u in test_map if u in cf.user_to_idx and test_map[u]]
    if test_n:
        users = users[:test_n]
    elif len(users) > 500:
        users = list(np.random.choice(users, 500, replace=False))
    print(f"  Users: {len(users)}, Items: {train['item_id'].nunique()}")

    # CF baseline (deterministic, no checkpoint needed)
    cf_ndcg, cf_recall = [], []
    for u in users:
        gt = set(test_map[u])
        cands, _ = cf.get_candidates(u, K)
        cf_ndcg.append(ndcg_at_k(cands, gt, TOP_K))
        cf_recall.append(recall_at_k(cands, gt, K))
    cf_m, cf_lo, cf_hi = bootstrap_ci(cf_ndcg)
    rec_m, rec_lo, rec_hi = bootstrap_ci(cf_recall)
    print(f"  CF-Score: NDCG={cf_m:.4f} [{cf_lo:.4f},{cf_hi:.4f}]  "
          f"Recall={rec_m:.4f} [{rec_lo:.4f},{rec_hi:.4f}]")

    results = {
        "CF_Score": {
            "ndcg_mean": cf_m, "ci": [cf_lo, cf_hi],
            "recall_mean": rec_m, "recall_ci": [rec_lo, rec_hi],
            "n": len(users),
        }
    }

    # Each LLM model
    for m in models_to_run:
        if len(m) == 6:
            provider, key_env, base_url, model_id, price_in, price_out = m
            ckpt_tag, extra_body, m_max_tokens = model_id, None, 120
        else:
            (provider, key_env, base_url, model_id, price_in, price_out,
             ckpt_tag, extra_body, m_max_tokens) = m
        model_key = ckpt_tag  # distinguishes think vs nothink variants
        short     = ckpt_tag.split("/")[-1]

        # Resume from checkpoint
        done = load_checkpoint(model_key, ds_name)
        todo = [u for u in users if str(u) not in done]
        if done:
            print(f"\n  [{short}] Resuming: {len(done)} done, {len(todo)} remaining")
        else:
            print(f"\n  [{short}] Starting fresh: {len(users)} users")

        try:
            client = _make_client(provider, base_url, key_env)
        except ValueError as e:
            print(f"  SKIP {short}: {e}")
            continue

        total_in_tokens, total_out_tokens = 0, 0
        llm_ndcg_all = {rec["user_id"]: rec["ndcg"] for rec in done.values()}
        req_delay = PROVIDER_DELAY.get(provider, 0.3)

        with tqdm(todo, desc=f"  {short}", leave=False) as pbar:
            for u in pbar:
                gt    = set(test_map[u])
                cands, _ = cf.get_candidates(u, K)
                hist  = [item_texts.get(x, f"Item {x}") for x in cf.history.get(u, [])[-10:]]
                cands_txt = [(c, item_texts.get(c, f"Item {c}")) for c in cands]

                history_str   = "\n".join(f"{i+1}. {t}" for i, t in enumerate(hist))
                candidates_str = "\n".join(
                    f"{i+1}. {t[:60]}" for i, (c, t) in enumerate(cands_txt[:30]))
                prompt = PROMPT.format(
                    history=history_str, candidates=candidates_str, top_k=TOP_K)

                t0 = time.time()
                resp, n_in, n_out = call_llm(client, model_id, prompt,
                                              max_tokens=m_max_tokens,
                                              extra_body=extra_body)
                latency = time.time() - t0
                # Throttle to respect provider rate limits
                elapsed = time.time() - t0
                if elapsed < req_delay:
                    time.sleep(req_delay - elapsed)

                n_shown = len(cands_txt)  # number of candidates shown (≤30)
                order  = parse_ranking(resp, n_shown)
                ranked = [cands_txt[i][0] for i in order]
                user_ndcg = ndcg_at_k(ranked, gt, TOP_K)
                llm_ndcg_all[str(u)] = user_ndcg
                total_in_tokens  += n_in
                total_out_tokens += n_out

                record = {
                    "user_id": str(u),
                    "ndcg": user_ndcg,
                    "n_in": n_in,
                    "n_out": n_out,
                    "latency_s": round(latency, 2),
                    # Store LLM ranking for Kendall's tau analysis (vs CF order 0,1,...,n-1)
                    "llm_order": order[:n_shown],
                }
                append_checkpoint(model_key, ds_name, record)
                pbar.set_postfix(ndcg=f"{user_ndcg:.4f}", lat=f"{latency:.1f}s")

        # Aggregate
        ndcg_vals = [llm_ndcg_all[str(u)] for u in users]
        m, lo, hi = bootstrap_ci(ndcg_vals)
        try:
            _, w_p = wilcoxon(cf_ndcg, ndcg_vals, zero_method="wilcox")
        except Exception:
            w_p = 1.0
        std_vals = np.std(ndcg_vals)
        d_cohen = (m - cf_m) / std_vals if std_vals > 1e-9 else 0.0
        total_cost = (total_in_tokens * price_in + total_out_tokens * price_out) / 1e6

        # Kendall's tau: agreement between CF order (0,1,...) and LLM order
        tau_vals = []
        if HAS_SCIPY_TAU:
            ckpt_data = load_checkpoint(model_key, ds_name)
            for u in users:
                rec = ckpt_data.get(str(u), {})
                llm_ord = rec.get("llm_order")
                if llm_ord and len(llm_ord) > 1:
                    n_c = len(llm_ord)
                    cf_ref = list(range(n_c))
                    tau, _ = _kendalltau(cf_ref, llm_ord)
                    if not np.isnan(tau):
                        tau_vals.append(tau)
        mean_tau = float(np.mean(tau_vals)) if tau_vals else None

        results[short] = {
            "provider": provider,
            "model_id": model_id,
            "ndcg_mean": m,
            "ci": [lo, hi],
            "vs_cf_pct": (m - cf_m) / (cf_m + 1e-9) * 100,
            "cohens_d": float(d_cohen),
            "wilcoxon_p": float(w_p),
            "mean_kendall_tau": mean_tau,
            "n": len(users),
            "total_in_tokens": total_in_tokens,
            "total_out_tokens": total_out_tokens,
            "cost_usd": round(total_cost, 4),
            "price_per_M": {"in": price_in, "out": price_out},
        }
        sig = "**SIG**" if w_p < 0.05 else "n.s."
        print(f"  [{short}] NDCG={m:.4f} [{lo:.4f},{hi:.4f}]  "
              f"vs CF={m-cf_m:+.4f} ({(m-cf_m)/(cf_m+1e-9)*100:+.1f}%)  "
              f"p={w_p:.3f} {sig}  cost=${total_cost:.3f}")

    return results


def print_summary(all_results: dict):
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    for ds_name, results in all_results.items():
        print(f"\n  {ds_name.upper()}")
        cf_m = results["CF_Score"]["ndcg_mean"]
        print(f"  {'Method':<30} {'NDCG':>8} {'vs CF':>8} {'d':>7} {'p':>7} {'$':>7}")
        print(f"  {'-'*62}")
        for model, r in results.items():
            pct  = r.get("vs_cf_pct", 0)
            d    = r.get("cohens_d", 0)
            p    = r.get("wilcoxon_p", 1)
            cost = r.get("cost_usd", 0)
            sig  = "*" if p < 0.05 else ""
            print(f"  {model:<30} {r['ndcg_mean']:>8.4f} {pct:>+7.1f}% "
                  f"{d:>7.3f} {p:>7.3f}{sig} {cost:>6.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_n", type=int, default=None,
                        help="Run on first N users only (pilot mode)")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        help="Which datasets to run")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model short names to run (default: all)")
    args = parser.parse_args()

    # Filter models if requested. Match against ckpt_tag (m[6]) when present,
    # else against model_id (m[3]).
    models_to_run = MODELS
    if args.models:
        def _tag(m):
            return (m[6] if len(m) >= 9 else m[3]).split("/")[-1]
        models_to_run = [m for m in MODELS if _tag(m) in args.models]

    np.random.seed(SEED)

    all_results: dict = {}
    # Load existing partial results
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            all_results = json.load(f).get("results", {})

    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"[SKIP] Unknown dataset: {ds_name}")
            continue
        data_path = DATASETS[ds_name]
        if not (PROJECT / data_path).exists():
            print(f"[SKIP] {ds_name}: data not found at {data_path}")
            continue
        ds_results = run_dataset(ds_name, data_path, args.test_n, models_to_run)
        all_results[ds_name] = ds_results

        # Save after each dataset
        with open(OUT_FILE, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "n_users": args.test_n or 500, "K": K, "top_k": TOP_K,
                    "seed": SEED, "n_factors": N_FACTORS,
                    "mode": "test" if args.test_n else "full",
                },
                "results": all_results,
            }, f, indent=2)
        print(f"\n  Saved → {OUT_FILE}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
