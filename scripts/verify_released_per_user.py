#!/usr/bin/env python3
"""Verify that the per-user checkpoints shipped with the artifact reproduce the published aggregates.

For every whitelisted checkpoint file this script
  * loads the per-user records (last record per user wins, as the drivers' resume logic does),
  * checks user_id hygiene: no value present both as int and as str, no silent merges,
  * recomputes the mean of the relevant per-user field,
  * compares it with the aggregate stored in the published result JSON.

Cells known NOT to reproduce are declared in KNOWN_NONREPRODUCING with the reason; the script
exits non-zero only if a cell outside that list fails, so it catches regressions without hiding
the known discrepancies. It writes experiments/logs/per_user_manifest.json.

Score-aware prompting (Tab. 7, §6.5) stores permutations, not metrics. Verify it separately with
    python scripts/cikm_score_aware_prompting.py --offline_only --out /tmp/sa.json
and compare every field except tokens/* and cost_usd (no API calls are made offline).

Usage:  python scripts/verify_released_per_user.py
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "experiments/logs"
CK = LOG / "checkpoints"
TOL = 5e-5

OLD_LLM = {"gpt-4o-mini": "gpt-4o-mini", "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
           "qwen3-32b": "qwen_qwen3-32b", "deepseek-chat": "deepseek-chat",
           "kimi-k2-turbo-preview": "kimi-k2-turbo-preview"}
KNOWN_NONREPRODUCING = {
    "multiapi_deepseek-v4-pro-think_movies.jsonl":
        "checkpoint was repaired after aggregation (154 corrupt records; see .bak_corrupt154 locally). "
        "It reproduces the post-repair value in rebuttal_multiapi_v4_testrun_20260511.json (-42.4%, p=.005), "
        "not the pre-repair value quoted in the paper (-43%, p=.004).",
    "multiapi_deepseek-v4-pro-nothink_movies.jsonl":
        "checkpoint mean .00894 vs published .00912 (paper: -7%, p=.245); cause not identified.",
}


def load(name):
    raw, recs = [], {}
    for line in (CK / name).open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        raw.append(r["user_id"])
        recs[str(r["user_id"])] = r
    ints = {str(u) for u in raw if isinstance(u, int)}
    strs = {u for u in raw if isinstance(u, str)}
    return recs, {"raw_records": len(raw), "unique_users": len(recs),
                  "user_id_types": sorted({type(u).__name__ for u in raw}),
                  "int_str_collisions": len(ints & strs)}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cells = []
    orig = json.loads((LOG / "rebuttal_multiapi_n500.json").read_text())["results"]
    testrun = json.loads((LOG / "rebuttal_multiapi_v4_testrun_20260511.json").read_text())["results"]
    v4 = json.loads((LOG / "v4_aggregate_n500.json").read_text())["datasets"]

    for ds, models in orig.items():
        for m, v in models.items():
            if m in OLD_LLM:
                cells.append((f"multiapi_{OLD_LLM[m]}_{ds}.jsonl", "§6.2" + ("; Tab. 5" if ds == "beauty" else ""),
                              "ndcg", v["ndcg_mean"], "rebuttal_multiapi_n500.json"))
    for ds, dv in v4.items():
        for m, v in dv["models"].items():
            if not m.startswith("deepseek-v4"):
                continue
            if ds == "beauty" and m.endswith("-think"):
                pub, src = testrun["beauty"][m]["ndcg_mean"], "rebuttal_multiapi_v4_testrun_20260511.json (n=200)"
            else:
                pub, src = v["ndcg10_mean"], "v4_aggregate_n500.json"
            cells.append((f"multiapi_{m}_{ds}.jsonl", "§6.3", "ndcg", pub, src))
    ks = json.loads((LOG / "cikm_k_sensitivity_n500.json").read_text())["results"]
    for K, v in ks.items():
        for pub_f, ck_f in [("llm_ndcg", "ndcg_llm"), ("cf_ndcg", "ndcg_cf"), ("recall_mean", "recall")]:
            cells.append((f"cikm_ksens_K{K}_beauty.jsonl", "Tab. 8", ck_f, v[pub_f], "cikm_k_sensitivity_n500.json"))
    ak = json.loads((LOG / "cikm_adaptive_k_ndcg.json").read_text())["results"]
    for ds, v in ak.items():
        for pub, f in [(v["recall"]["fixed_K100"]["mean"], "fk_recall"), (v["recall"]["adaptive_K"]["mean"], "ak_recall"),
                       (v["ndcg_llm"]["fixed_K100"]["mean"], "fk_ndcg_llm"), (v["ndcg_llm"]["adaptive_K"]["mean"], "ak_ndcg_llm")]:
            cells.append((f"cikm_adaptiveK_{ds}.jsonl", "§7.2 adaptive-K", f, pub, "cikm_adaptive_k_ndcg.json"))

    files, unexpected = {}, []
    for fname, item, field, pub, src in cells:
        if fname not in files:
            if not (CK / fname).exists():
                files[fname] = {"file": fname, "missing": True}
                unexpected.append(f"{fname}: missing"); continue
            recs, hyg = load(fname)
            files[fname] = {"file": fname, "paper_items": set(), "sha256": sha(CK / fname), **hyg,
                            "checks": [], "_recs": recs}
            if hyg["int_str_collisions"]:
                unexpected.append(f"{fname}: {hyg['int_str_collisions']} user_id int/str collisions")
        f = files[fname]
        if f.get("missing"):
            continue
        f["paper_items"].add(item)
        got = float(np.mean([r[field] for r in f["_recs"].values()]))
        ok = abs(got - pub) <= TOL
        f["checks"].append({"field": field, "published": pub, "recomputed": got, "source": src, "reproduces": ok})
        if not ok and fname not in KNOWN_NONREPRODUCING:
            unexpected.append(f"{fname} {field}: published {pub:.5f} vs recomputed {got:.5f}")

    for fname in sorted(CK.glob("scoreaware_*.jsonl")):
        recs, hyg = load(fname.name)
        files[fname.name] = {"file": fname.name, "paper_items": {"Tab. 7", "§6.5"}, "sha256": sha(fname), **hyg,
                             "checks": [{"field": "llm_order (permutation)", "reproduces": None,
                                         "source": "verify with cikm_score_aware_prompting.py --offline_only (2026-09-17: all 247 metric fields identical)"}]}

    manifest = []
    for f in files.values():
        f.pop("_recs", None)
        f["paper_items"] = sorted(f.get("paper_items", []))
        if f["file"] in KNOWN_NONREPRODUCING:
            f["known_discrepancy"] = KNOWN_NONREPRODUCING[f["file"]]
        manifest.append(f)
    manifest.sort(key=lambda x: x["file"])
    (LOG / "per_user_manifest.json").write_text(json.dumps(
        {"generated_by": "scripts/verify_released_per_user.py", "tolerance": TOL, "files": manifest}, indent=1))

    n_ok = sum(c["reproduces"] is True for f in manifest for c in f.get("checks", []))
    n_bad = sum(c["reproduces"] is False for f in manifest for c in f.get("checks", []))
    print(f"{len(manifest)} files, {n_ok} checks reproduce, {n_bad} do not "
          f"({sum(1 for f in manifest if 'known_discrepancy' in f)} files with declared discrepancies)")
    for f in manifest:
        if "known_discrepancy" in f:
            print(f"  known: {f['file']}: {f['known_discrepancy'][:110]}")
    if unexpected:
        print("\nUNEXPECTED:"); [print("  " + u) for u in unexpected]; sys.exit(1)
    print("no unexpected discrepancies")


if __name__ == "__main__":
    main()
