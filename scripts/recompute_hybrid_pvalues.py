#!/usr/bin/env python3
"""Recompute the null p-values in cikm_hybrid_retrieval_v4.json from its stored per-user arrays.

The driver called `_, p = safe_wilcoxon(...)`; safe_wilcoxon returns a bare float, so the
unpack always raised and every `p_vs_CF_SVD` was written as null. The per-user Recall@100
arrays were saved correctly, so the p-values can be recovered without re-running retrieval
(the Dense channel would need a GPU). Before writing, the script checks that each stored
mean equals the mean of its per-user array.

Usage:  python scripts/recompute_hybrid_pvalues.py [--out PATH]   # default: rewrite in place
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cikm_controlled_recall_sweep import safe_wilcoxon  # noqa: E402

SRC = ROOT / "experiments/logs/cikm_hybrid_retrieval_v4.json"
METHODS = ["BM25", "Dense", "RRF-3way", "RRF-CFBM25", "RRF-CFDense"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(SRC)); a = ap.parse_args()
    d = json.loads(SRC.read_text())
    print(f"{'dataset':<28}{'method':<13}{'mean':>9}{'check':>7}{'p_vs_CF_SVD':>13}")
    for ds, v in d.items():
        if not isinstance(v, dict) or "per_user" not in v:
            continue
        pu, res = v["per_user"], v["results"]
        for name, arr in pu.items():
            ok = abs(float(np.mean(arr)) - res[name]["recall_at_100_mean"]) < 1e-12
            if not ok:
                sys.exit(f"{ds}/{name}: stored mean does not match per-user array; refusing to write")
        cf = np.array(pu["CF-SVD"], dtype=float)
        for name in METHODS:
            p = safe_wilcoxon(np.array(pu[name], dtype=float), cf)
            r = res[name]
            r.pop("p_vs_CF_SVD_note", None)
            if p is None or np.isnan(p):
                r["p_vs_CF_SVD"] = None
                r["p_vs_CF_SVD_note"] = "not testable: all paired differences zero or fewer than 5 nonzero"
                shown = "n/a"
            else:
                r["p_vs_CF_SVD"] = float(p); shown = f"{p:.4f}"
            print(f"{ds:<28}{name:<13}{r['recall_at_100_mean']:>9.4f}{'ok':>7}{shown:>13}")
    d["_p_value_recompute"] = ("p_vs_CF_SVD recomputed 2026-09-17 by scripts/recompute_hybrid_pvalues.py from the stored "
                               "per-user arrays; the original run nulled them through a tuple-unpack bug. Recall values unchanged.")
    Path(a.out).write_text(json.dumps(d, indent=2))
    print(f"\n-> {a.out}")

if __name__ == "__main__":
    main()
