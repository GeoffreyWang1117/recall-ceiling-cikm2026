#!/usr/bin/env python3
"""
Decompose the CIKM "injection-oracle" protocol into its separable components and
measure which component moves the SELECTION among rerankers.

The published oracle arm (kdd_oracle_realistic_n500.py) does three things at once:
    (i)   inserts every unretrieved positive
    (ii)  drops CF ranks 91-100
    (iii) shuffles the whole candidate list
so no component can be attributed. This script separates them.

Critical constraint: the listwise LLM sees only the first `--window` (30)
candidates. Inserting a positive at slot 100 would be invisible to it -- which is
exactly why the published oracle arm had to shuffle. All insertion positions here
are therefore defined INSIDE the visible window.

Arms (every arm yields exactly K=100 candidates; visible window = first 30):
    R            CF ranks 1..100, CF order                      (baseline)
    drop30       slot 30 <- CF rank 101                         (displacement only)
    gold30       slot 30 <- an unretrieved positive             (MAIN intervention)
    gold01       slot 1  <- that positive, CF rank 30 removed   (position factor)
    shuffle      CF ranks 1..100, visible window shuffled       (order only)
    cikm_oracle  ranks 1..90 + all unretrieved positives, cut to 100, shuffled

Key contrasts:
    R       -> drop30   effect of displacing one window-tail candidate
    drop30  -> gold30   effect of the positive, holding displacement fixed  <-- key
    gold30  -> gold01   position sensitivity
    R       -> shuffle  order effect
    cikm_oracle         does the compound equal the sum of its parts?

Modes:
    --mode bound      Corollary 3 ceiling recomputation. No LLM calls, deterministic.
    --mode generate   run rerankers, append per-user JSONL per (method, arm)
    --mode score      offline scoring + selection analysis from checkpoints

Records store the PERMUTATION, not the metric, so the metric convention can be
changed offline without re-running anything.
"""
import argparse, json, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Seeded CFSVD (random_state=42) + set-based multi-positive NDCG.
# NEVER import these from a kdd_* script: 17 of them build TruncatedSVD with no
# random_state, which makes their candidate sets irreproducible run to run.
from cikm_controlled_recall_sweep import CFSVD, ndcg_at_k, safe_wilcoxon, bootstrap_ci  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"
CKPT_DIR = PROJECT_ROOT / "experiments" / "logs" / "checkpoints"
OUT_DIR = PROJECT_ROOT / "experiments" / "logs"

SEED = 42
K = 100
WINDOW = 30
TOP_K = 10

ARMS = ["R", "drop30", "gold30", "gold01", "shuffle", "cikm_oracle"]
FREE_METHODS = ["cf-score", "as-presented"]
# mistral:7b removed 2026-09-11: it does not respond at all, even standalone with a
# 300 s timeout on a trivial prompt, while the GPU sits idle. Broken blob or a
# version mismatch; re-pulling is a separate decision.
LLM_METHODS = ["qwen2.5:0.5b", "qwen2.5:3b", "qwen2.5:7b", "llama3.1:8b"]

PROMPT = """You are a recommendation system. Based on the user's purchase history, \
rank the candidate products from most to least likely to be purchased next.

User's recent purchases:
{history}

Candidate products:
{candidates}

Return ONLY the top {top_k} candidate numbers, most likely first, comma-separated. \
No explanation."""

_ckpt_lock = threading.Lock()


# ── data ─────────────────────────────────────────────────────────────────────

def load_dataset(name):
    d = PROJECT_ROOT / "data" / "processed" / name
    train = pd.read_parquet(d / "train.parquet")
    test = pd.read_parquet(d / "test.parquet")
    meta = json.loads((d / "item_metadata.json").read_text())
    def text(i):
        v = meta.get(str(i))
        if isinstance(v, dict):
            return v.get("text") or v.get("title") or f"Product {i}"
        return v or f"Product {i}"
    return train, test, text


def build_context(train, test, n_users, seed=SEED):
    """CF retrieval + all six candidate sets per user. Deterministic."""
    cf = CFSVD(n_factors=128)
    cf.fit(train)
    history = train.groupby("user_id")["item_id"].apply(list).to_dict()
    gt_map = test.groupby("user_id")["item_id"].apply(list).to_dict()

    valid = sorted(u for u in gt_map if u in cf.user_to_idx and len(gt_map[u]) > 0)
    rng = np.random.default_rng(seed)
    users = sorted(rng.choice(valid, min(n_users, len(valid)), replace=False).tolist()) \
        if len(valid) > n_users else valid

    ctx = {}
    for uid in users:
        hist = history.get(uid, [])
        cands, _ = cf.topk(uid, K + 1, exclude_ids=hist)
        if len(cands) < K + 1:
            continue
        base = cands[:K]
        filler = cands[K]                      # CF rank 101
        gt = set(gt_map[uid])
        hist_set = set(hist)
        retrieved = gt & set(base)
        unretrieved = sorted(gt - set(base))
        # A positive that is also in the TRAINING HISTORY was excluded by
        # retrieval on purpose (repeat purchase), not missed. Inserting it is a
        # different intervention: it carries a top-ranked raw CF score, so any
        # score-respecting method promotes it to rank 1 and the arm shows a huge
        # artificial gain. 12.5% of Beauty users have such a positive. The clean
        # single-component arms therefore draw gold from the history-free pool.
        insertable = sorted(gt - set(base) - hist_set)

        # deterministic per-user RNG so arms are reproducible independently
        r = random.Random(f"{seed}-{uid}")
        gold = r.choice(insertable) if insertable else None
        gold_inserted = gold is not None
        ins = gold if gold_inserted else filler       # fall back to filler

        arms = {}
        arms["R"] = list(base)
        arms["drop30"] = base[:WINDOW - 1] + [filler] + base[WINDOW:]
        arms["gold30"] = base[:WINDOW - 1] + [ins] + base[WINDOW:]
        arms["gold01"] = [ins] + base[:WINDOW - 1] + base[WINDOW:]
        win = base[:WINDOW][:]
        r.shuffle(win)
        arms["shuffle"] = win + base[WINDOW:]
        orc = base[:90] + [g for g in unretrieved if g not in base[:90]]
        orc = orc[:K]
        if len(orc) < K:
            orc += [c for c in base[90:] if c not in orc][:K - len(orc)]
        r.shuffle(orc)
        arms["cikm_oracle"] = orc

        for a, c in arms.items():
            assert len(c) == K, f"{uid}/{a}: {len(c)} != {K}"
        if gold is not None:
            assert gold not in hist_set, f"{uid}: gold is a training-history item"
            assert gold in gt and gold not in set(base), f"{uid}: bad gold"

        # raw CF score for every item that can appear in any arm
        uvec = cf.user_factors[cf.user_to_idx[uid]]
        allitems = set().union(*[set(v) for v in arms.values()])
        sc = {}
        for it in allitems:
            j = cf.item_to_idx.get(it)
            if j is None or it in hist_set:
                sc[it] = -1e9          # unseen item, or masked by retrieval
            else:
                sc[it] = float(cf.item_factors[j] @ uvec)

        ctx[uid] = {
            "user_id": str(uid),
            "history": hist,
            "gt": sorted(gt),
            "n_positives": len(gt),
            "m_retrieved": len(retrieved),
            "recall_at_K": len(retrieved) / len(gt),
            "gold_item": gold,
            "gold_inserted": gold_inserted,
            "filler_item": filler,
            "arms": arms,
            "cf_scores": sc,
        }
    return cf, ctx


# ── ceiling (Corollary 3) ────────────────────────────────────────────────────

def idcg(m, k=TOP_K):
    return sum(1.0 / np.log2(i + 2) for i in range(min(m, k)))


def mode_bound(ctx, dataset):
    """Corollary 3: E[IDCG_k(|W∩Y|)/IDCG_k(|Y|)] vs the Recall@K the paper quotes."""
    rows = []
    for u, c in ctx.items():
        m, ny = c["m_retrieved"], c["n_positives"]
        rows.append({
            "user_id": c["user_id"], "n_positives": ny, "m_retrieved": m,
            "recall": c["recall_at_K"],
            "cor2": (idcg(m) / idcg(ny)) if ny > 0 else 0.0,
        })
    df = pd.DataFrame(rows)
    rec_m, rec_lo, rec_hi = bootstrap_ci(df["recall"].tolist(), n=10000, seed=SEED)
    c2_m, c2_lo, c2_hi = bootstrap_ci(df["cor2"].tolist(), n=10000, seed=SEED)
    out = {
        "dataset": dataset, "n_users": len(df), "K": K, "top_k": TOP_K, "seed": SEED,
        "note": "New deterministic computation (CFSVD random_state=42). NOT a recovery "
                "of the original Table 3 run, whose CF-SVD was unseeded.",
        "positives_per_user": {"mean": float(df.n_positives.mean()),
                               "max": int(df.n_positives.max()),
                               "frac_exactly_1": float((df.n_positives == 1).mean())},
        "recall_at_K": {"mean": rec_m, "ci_low": rec_lo, "ci_high": rec_hi},
        "corollary2_bound": {"mean": c2_m, "ci_low": c2_lo, "ci_high": c2_hi},
        "ratio_cor2_over_recall": (c2_m / rec_m) if rec_m else None,
        "per_user": df.to_dict(orient="list"),
    }
    p = OUT_DIR / f"cikm_corollary2_bound_{dataset}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  users                     {len(df)}")
    print(f"  positives/user            mean {df.n_positives.mean():.3f}, "
          f"max {int(df.n_positives.max())}, exactly-1 {100*(df.n_positives==1).mean():.1f}%")
    print(f"  Recall@{K} (paper's bound) {rec_m:.4f}  [{rec_lo:.4f}, {rec_hi:.4f}]")
    print(f"  Corollary 3 bound         {c2_m:.4f}  [{c2_lo:.4f}, {c2_hi:.4f}]")
    print(f"  ratio Cor2 / Recall       {c2_m/rec_m:.2f}x" if rec_m else "")
    print(f"\n  -> {p}")
    return out


# ── rerankers ────────────────────────────────────────────────────────────────

def parse_ranking(resp, n):
    """Return (indices, n_parsed). Unlike the original, the caller can see how
    many positions the model actually supplied."""
    import re
    idx, seen = [], set()
    for tok in re.findall(r"\d+", resp):
        i = int(tok) - 1
        if 0 <= i < n and i not in seen:
            idx.append(i); seen.add(i)
    n_parsed = len(idx)
    for i in range(n):
        if i not in seen:
            idx.append(i)
    return idx[:n], n_parsed


def call_ollama(model, prompt, timeout=180):
    r = requests.post(OLLAMA_URL, json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "stream": False, "options": {"temperature": 0.1, "num_predict": 300, "seed": SEED},
    }, timeout=timeout)
    return r.json()["message"]["content"]


def ckpt_path(method, arm, dataset):
    safe = method.replace("/", "_").replace(":", "_")
    return CKPT_DIR / f"armdecomp_{safe}_{arm}_{dataset}.jsonl"


def load_ckpt(method, arm, dataset):
    p = ckpt_path(method, arm, dataset)
    if not p.exists():
        return {}
    out = {}
    for line in p.open():
        try:
            r = json.loads(line)
            out[str(r["user_id"])] = r
        except Exception:
            continue
    return out


def append_ckpt(method, arm, dataset, rec):
    with _ckpt_lock:
        with ckpt_path(method, arm, dataset).open("a") as f:
            f.write(json.dumps(rec) + "\n")


def run_one(method, arm, dataset, c, text_fn):
    cands = c["arms"][arm]
    vis = cands[:WINDOW]
    t0 = time.time()
    hist_t = "\n".join(f"{i+1}. {text_fn(t)}" for i, t in enumerate(c["history"][-10:]))
    cand_t = "\n".join(f"{i+1}. Item {i+1}: {text_fn(it)}" for i, it in enumerate(vis))
    try:
        resp = call_ollama(method, PROMPT.format(history=hist_t, candidates=cand_t, top_k=TOP_K))
        status = "ok" if resp.strip() else "api_empty"
    except Exception as e:
        resp, status = "", f"api_fail:{type(e).__name__}"
    order, n_parsed = parse_ranking(resp, len(vis))
    if status == "ok" and n_parsed == 0:
        status = "parse_empty"
    elif status == "ok" and n_parsed < TOP_K:
        status = "parse_partial"
    return {
        "user_id": c["user_id"], "arm": arm, "method": method,
        "llm_order": order, "status": status, "n_parsed": n_parsed,
        "n_in": len(cand_t), "n_out": len(resp), "latency_s": round(time.time() - t0, 3),
    }


def mode_generate(ctx, dataset, text_fn, methods, arms, workers):
    for method in methods:
        for arm in arms:
            done = load_ckpt(method, arm, dataset)
            # A transport failure must NOT count as done, or the user keeps its
            # identity-fallback record forever. Re-appending on retry is safe:
            # load_ckpt keeps the last occurrence per user.
            todo = [c for u, c in ctx.items()
                    if c["user_id"] not in done
                    or str(done[c["user_id"]].get("status", "")).startswith("api_")]
            if not todo:
                print(f"  [{method} | {arm}] complete ({len(done)})"); continue
            t0 = time.time()
            print(f"  [{method} | {arm}] {len(todo)} to run "
                  f"({len(done)} cached) ...", flush=True)
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for rec in ex.map(lambda c: run_one(method, arm, dataset, c, text_fn), todo):
                        append_ckpt(method, arm, dataset, rec)
            else:
                for c in todo:
                    append_ckpt(method, arm, dataset, run_one(method, arm, dataset, c, text_fn))
            el = time.time() - t0
            print(f"      done in {el/60:.1f} min ({el/max(len(todo),1):.2f} s/call)")


# ── scoring ──────────────────────────────────────────────────────────────────

def status_tier(rec):
    """Quality tier re-derived from the stored n_parsed, so existing checkpoints
    can be re-classified offline without re-running anything. The generation-time
    label is only trusted for transport failures."""
    s = rec.get("status", "")
    if s.startswith("api_"):
        return s
    n = rec.get("n_parsed", 0)
    if n == 0:
        return "parse_empty"       # ranking is entirely the identity fallback
    if n < 3:
        return "degenerate"        # essentially the identity fallback
    if n < TOP_K:
        return "short"             # usable: model supplied a partial ranking
    return "ok"


def final_list(c, arm, order=None):
    """Closed-candidate output: reranked window, remainder appended in given order."""
    cands = c["arms"][arm]
    vis, rest = cands[:WINDOW], cands[WINDOW:]
    if order is None:
        return list(cands)
    return [vis[i] for i in order] + rest


def score_free(method, c, arm):
    if method == "as-presented":
        return final_list(c, arm)
    if method == "cf-score":
        cands = c["arms"][arm]
        sc = c["cf_scores"]
        return sorted(cands, key=lambda it: -sc.get(it, -1e9))
    raise ValueError(method)


def mode_score(ctx, dataset, methods, arms, out_name):
    users = [c["user_id"] for c in ctx.values()]
    by_uid = {c["user_id"]: c for c in ctx.values()}
    per_user, status_counts, incomplete = {}, {}, []

    for method in methods:
        for arm in arms:
            key = f"{method}|{arm}"
            if method in FREE_METHODS:
                per_user[key] = [ndcg_at_k(score_free(method, by_uid[u], arm),
                                           by_uid[u]["gt"], TOP_K) for u in users]
                continue
            ck = load_ckpt(method, arm, dataset)
            missing = [u for u in users if u not in ck]
            if missing:
                incomplete.append((key, len(missing)))
                continue
            arr, sc = [], {}
            for u in users:
                r = ck[u]
                t = status_tier(r)
                sc[t] = sc.get(t, 0) + 1
                arr.append(ndcg_at_k(final_list(by_uid[u], arm, r["llm_order"]),
                                     by_uid[u]["gt"], TOP_K))
            per_user[key] = arr
            status_counts[key] = sc

    if incomplete:
        print("\n  INCOMPLETE -- refusing to score (paired analysis needs equal user sets):")
        for k, n in incomplete:
            print(f"    {k}: {n} users missing")
        print("  Run --mode generate first.")
        return None

    print(f"\n  {'method':<16}" + "".join(f"{a:>13}" for a in arms))
    for method in methods:
        row = f"  {method:<16}"
        for arm in arms:
            row += f"{np.mean(per_user[f'{method}|{arm}']):>13.5f}"
        print(row)

    unusable = {k: {s: c for s, c in v.items() if s in ("parse_empty", "degenerate")
                    or s.startswith("api_")}
                for k, v in status_counts.items()}
    unusable = {k: v for k, v in unusable.items() if v}
    print(f"\n  status tiers (first arm shown): "
          f"{next(iter(status_counts.values())) if status_counts else 'n/a'}")
    print(f"  UNUSABLE (identity-fallback) records: {unusable if unusable else 'none'}")

    out = {
        "dataset": dataset, "n_users": len(users), "K": K, "window": WINDOW,
        "top_k": TOP_K, "seed": SEED, "arms": arms, "methods": methods,
        "aggregate": {k: {"ndcg_mean": float(np.mean(v))} for k, v in per_user.items()},
        "status_counts": status_counts,
        "gold_inserted_frac": float(np.mean([by_uid[u]["gold_inserted"] for u in users])),
        "users": users,
        "per_user": {k: [float(x) for x in v] for k, v in per_user.items()},
    }
    p = OUT_DIR / out_name
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  -> {p}")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bound", "generate", "score"], required=True)
    ap.add_argument("--dataset", default="amazon_beauty_sampled")
    ap.add_argument("--n-users", type=int, default=500)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only the first N users OF THE SAME n-users sample; "
                         "smoke-test records are then reused by the full run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, text_fn = load_dataset(a.dataset)
    print(f"{a.dataset}: train={len(train)} test={len(test)}")
    cf, ctx = build_context(train, test, a.n_users)
    if a.limit:
        ctx = {k: v for k, v in list(ctx.items())[:a.limit]}
    print(f"context built for {len(ctx)} users "
          f"(gold insertable for {sum(c['gold_inserted'] for c in ctx.values())})")

    if a.mode == "bound":
        mode_bound(ctx, a.dataset); return

    methods = a.methods if a.methods is not None else (FREE_METHODS + LLM_METHODS)
    if a.mode == "generate":
        mode_generate(ctx, a.dataset, text_fn,
                      [m for m in methods if m not in FREE_METHODS], a.arms, a.workers)
    else:
        mode_score(ctx, a.dataset, methods, a.arms,
                   a.out or f"cikm_arm_decomposition_{a.dataset}.json")


if __name__ == "__main__":
    main()
