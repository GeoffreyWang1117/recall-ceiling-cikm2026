#!/usr/bin/env python3
"""Recompute ceiling utilisation (eta) against the Corollary 3 bound.

The camera-ready Table 5 and Figure 1C divide each method's NDCG@10 by
Recall@|W|. That is the leave-one-out (Corollary 2) bound, but the shipped test
splits are multi-positive, where Recall@K is not an upper bound on NDCG@k. The
applicable bound is Corollary 3:  E[ IDCG_k(|W∩Y_u|) / IDCG_k(|Y_u|) ].

Numerators are read unchanged from cikm_eta_windows.json. Denominators are
recomputed on the same user sample with seeded CFSVD. The script refuses to
write anything unless its Recall@30 and Recall@100 match the ones stored in
cikm_eta_windows.json, which is the check that numerator and denominator come
from the same users.

Naming: identifiers such as `cor2`, `corollary2_bound` and the output file name predate the
check of the compiled paper and are kept for compatibility. They refer to the paper's
Corollary 3 (multi-positive specialisation); Corollary 2 in the paper is the LOO case.
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cikm_arm_decomposition import load_dataset, build_context, idcg, TOP_K  # noqa: E402

SRC = ROOT / "experiments/logs/cikm_eta_windows.json"
OUT = ROOT / "experiments/logs/cikm_eta_windows_cor2.json"


def main():
    src = json.loads(SRC.read_text())
    train, test, _ = load_dataset("amazon_beauty_sampled")
    _, ctx = build_context(train, test, src["n_users"])
    rows = []
    for c in ctx.values():
        gt = set(c["gt"]); base = c["arms"]["R"]; ny = len(gt)
        m30, m100 = len(gt & set(base[:30])), len(gt & set(base))
        rows.append((m30 / ny, m100 / ny, idcg(m30) / idcg(ny), idcg(m100) / idcg(ny)))
    r30, r100, c30, c100 = (float(np.mean([r[i] for r in rows])) for i in range(4))

    ok30 = abs(r30 - src["recall_at_30"]) < 1e-12
    ok100 = abs(r100 - src["recall_at_100"]) < 1e-12
    print(f"Recall@30  recomputed {r30:.10f}  stored {src['recall_at_30']:.10f}  match={ok30}")
    print(f"Recall@100 recomputed {r100:.10f}  stored {src['recall_at_100']:.10f}  match={ok100}")
    if not (ok30 and ok100):
        sys.exit("user sample differs from cikm_eta_windows.json; refusing to write")

    denom = {30: c30, 100: c100}
    methods = []
    for m in src["methods"]:
        methods.append({**m, "cor2_bound_own_window": denom[m["window"]],
                        "eta_vs_cor2_own_window": m["ndcg"] / denom[m["window"]]})
    out = {"source": SRC.name, "dataset": src["dataset"], "n_users": len(rows),
           "recall_at_30": r30, "recall_at_100": r100,
           "cor2_bound_at_30": c30, "cor2_bound_at_100": c100,
           "note": "eta_vs_cor2_own_window = ndcg / bound of the paper's Corollary 3 (multi-positive) at the method's own window",
           "methods": methods}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nCor2 bound @30 {c30:.6f}   @100 {c100:.6f}")
    print(f"{'method':<16}{'|W|':>5}{'eta vs Recall':>15}{'eta vs Cor2':>13}")
    for m in methods:
        print(f"{m['method']:<16}{m['window']:>5}{100*m['eta_vs_own_window']:>14.1f}%{100*m['eta_vs_cor2_own_window']:>12.1f}%")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
