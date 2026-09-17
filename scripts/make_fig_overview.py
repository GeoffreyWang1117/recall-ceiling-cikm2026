#!/usr/bin/env python
"""Generate figures/fig_overview.png|pdf for the CIKM 2026 camera-ready.

The submitted version of this figure was drawn in an external tool and could
not be regenerated, which let three errors survive review:

  * panel A quoted "Recall@K = 2-17%" (the eight-dataset range is 2-19%) and
    bounded NDCG by Recall@K rather than by Recall@|W_pi|, which is what
    Theorem 1 actually says once the reranking window is defined;
  * panel B's injection-oracle bars did not match Table 2 (.084/.104/.086
    against the measured .0856/.1039/.0971) and omitted the closed-candidate
    bound entirely, which is the quantity the theorem contributes;
  * panel C printed ceiling-utilisation values computed against Recall@100 for
    every method, including the listwise LLMs that only ever saw a 30-candidate
    window (GPT-4o-mini 5.5%, DeepSeek-V3 2.0%, Llama-3.3 4.7%, LoRA 8.6%).

Every number below is read from the experiment JSONs rather than typed in, so
the figure cannot drift from the tables again.

Usage:  python scripts/make_fig_overview.py                  # camera-ready figure, unchanged
        python scripts/make_fig_overview.py --variant v2     # arXiv v2 figure

--variant v2 (added 2026-09-17) fixes a scope error the camera-ready figure
inherited: the shipped test splits are multi-positive, where Recall@K is NOT an
upper bound on NDCG@k (Corollary 2 needs |Y_u| = 1). In v2 the panel-B bound is
the Corollary 3 bound read from cikm_corollary2_bound_*.json, panel-C eta divides
by that bound (cikm_eta_windows_cor2.json), and panel A states Theorem 1 in its
general form. The default output is left byte-for-byte as submitted.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETA_JSON = os.path.join(REPO, "experiments", "logs", "cikm_eta_windows.json")
OUT_DIR = os.path.join(REPO, "submissions", "cikm2026", "figures")
V2_OUT_DIR = os.path.join(REPO, "submissions", "cikm2026", "arxiv_v2", "figures")
ETA_V2_JSON = os.path.join(REPO, "experiments", "logs", "cikm_eta_windows_cor2.json")
COR2_JSON = os.path.join(REPO, "experiments", "logs", "cikm_corollary2_bound_amazon_{}_sampled.json")

# Text that differs between variants. main() switches these for --variant v2.
CEILING_TEX = r"$\mathbb{E}[\mathrm{NDCG}@k]\ \leq\ \mathrm{Recall}@|W_\pi|$"
BOUND_LABEL = "closed-candidate bound (Thm. 1)"
ETA_XLABEL = "ceiling utilisation $\\eta$ = NDCG@10 / Recall@$|W_\\pi|$  (Beauty)"
ETA_KEY = "eta_vs_own_window"
# v2 only: the Corollary 3 bound is tall enough that the oracle->realistic gap
# arrow would cross the bound bar and its label. Draw the bound bar last so the
# arrow only joins adjacent bars.
BOUND_LAST = False

# ---------------------------------------------------------------------------
# Panel B: injection-oracle vs closed-candidate bound vs realistic (Table 2 and
# the bound stated in Section 5). Oracle/realistic are Gemma3-4B at n=500; the
# bound is Recall@100 on the canonical sample.
# ---------------------------------------------------------------------------
PANEL_B = {
    "Beauty":      {"oracle": 0.0856, "bound": 0.0823, "realistic": 0.0065},
    "Movies":      {"oracle": 0.1039, "bound": 0.0295, "realistic": 0.0067},
    "Electronics": {"oracle": 0.0971, "bound": 0.0224, "realistic": 0.0050},
}

# Panel C: which methods to show, in the order they should appear.
PANEL_C_ORDER = [
    "Qwen3-8B", "GPT-4o-mini", "Llama-3.3-70B", "DeepSeek-V3",
    "LambdaMART", "LoRA-LLaMA-3B", "MLP", "CF-SVD",
]
GROUP = {  # colour key
    "Qwen3-8B": "llm", "GPT-4o-mini": "llm", "Llama-3.3-70B": "llm",
    "DeepSeek-V3": "llm", "LambdaMART": "sup", "LoRA-LLaMA-3B": "sup",
    "MLP": "sup", "CF-SVD": "cf",
}
COLOR = {"llm": "#4C72B0", "sup": "#DD8452", "cf": "#8C8C8C"}

INK = "#1a1a1a"
RED = "#B22222"


def load_eta():
    with open(ETA_JSON) as fh:
        data = json.load(fh)
    eta = {}
    for m in data["methods"]:
        name = "CF-SVD" if m["method"] == "CF-Score" else m["method"]
        eta[name] = (100.0 * m[ETA_KEY], m["window"])
    return eta, data


def box(ax, x, y, w, h, text, sub=None, fc="white", ec=INK, fs=5.8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=0.7, edgecolor=ec, facecolor=fc, zorder=2))
    cy = y + h / 2 + (0.09 if sub else 0.0)
    ax.text(x + w / 2, cy, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.11, sub, ha="center", va="center",
                fontsize=fs - 0.9, color="#444444", zorder=3)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=5,
        linewidth=0.7, color=INK, zorder=4))


def panel_a(ax, recall_lo, recall_hi):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.90, "(A)", fontsize=7.4, fontweight="bold", color=INK)

    ax.text(0.53, 0.90,
            r"Recall ceiling (Theorem 1):  " + CEILING_TEX,
            ha="center", va="center", fontsize=7.2, color=RED,
            fontweight="bold")
    ax.plot([0.01, 0.99], [0.68, 0.68], linestyle=(0, (5, 3)),
            linewidth=1.0, color=RED)
    ax.text(0.99, 0.555, r"no reranker can recover an item absent from $C_u$",
            ha="right", va="center", fontsize=5.2, color="#555555")

    y, h = 0.03, 0.44
    widths = [0.155, 0.185, 0.235, 0.205, 0.105]
    gap = (1.0 - sum(widths)) / (len(widths) - 1) * 0.92
    total = sum(widths) + gap * (len(widths) - 1)
    xs, x = [], (1.0 - total) / 2
    for w in widths:
        xs.append(x)
        x += w + gap

    box(ax, xs[0], y, widths[0], h, r"user profile $X_u$")
    box(ax, xs[1], y, widths[1], h, r"retrieval $\mathcal{R}$",
        sub=f"Recall@$K$ = {recall_lo:.0f}–{recall_hi:.0f}%", fc="#DCE7F5")
    box(ax, xs[2], y, widths[2], h, r"candidates $C_u$,  $K{=}100$",
        sub=r"window $|W_\pi| \in \{30, 100\}$")
    box(ax, xs[3], y, widths[3], h, r"reranker $\pi$",
        sub="LLM / supervised / tuned", fc="#FBE3D0")
    box(ax, xs[4], y, widths[4], h, r"top-$k$")

    for i in range(4):
        arrow(ax, xs[i] + widths[i] + 0.004, xs[i + 1] - 0.004, y + h / 2)


def panel_b(ax):
    names = list(PANEL_B)
    xpos = range(len(names))
    w = 0.25
    if BOUND_LAST:
        series = [
            ("injection-oracle", "oracle", "#BFBFBF", -w),
            ("realistic retrieval", "realistic", "#B22222", 0.0),
            (BOUND_LABEL, "bound", "#7BA7D7", w),
        ]
    else:
        series = [
            ("injection-oracle", "oracle", "#BFBFBF", -w),
            (BOUND_LABEL, "bound", "#7BA7D7", 0.0),
            ("realistic retrieval", "realistic", "#B22222", w),
        ]
    real_off = 0.0 if BOUND_LAST else w
    for label, key, color, off in series:
        vals = [PANEL_B[n][key] for n in names]
        ax.bar([x + off for x in xpos], vals, width=w, color=color,
               edgecolor=INK, linewidth=0.5, label=label, zorder=2)
        for x, v in zip(xpos, vals):
            ax.text(x + off, v + 0.0025, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=4.4, color=INK)

    for x, n in zip(xpos, names):
        gap = 100 * (1 - PANEL_B[n]["realistic"] / PANEL_B[n]["oracle"])
        ax.annotate("", xy=(x + real_off, PANEL_B[n]["realistic"] + 0.008),
                    xytext=(x - w, PANEL_B[n]["oracle"] + 0.008),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8,
                                    connectionstyle="arc3,rad=-0.25"),
                    zorder=4)
        ax.text(x, PANEL_B[n]["oracle"] + 0.021, f"−{gap:.1f}%",
                ha="center", va="bottom", fontsize=5.6, color=RED,
                fontweight="bold")

    ax.set_xticks(list(xpos))
    ax.set_xticklabels(names, fontsize=6.2)
    ax.set_ylabel("NDCG@10", fontsize=6.4, labelpad=1.5)
    ax.set_ylim(0, 0.175)
    ax.set_yticks([0.00, 0.04, 0.08, 0.12])
    ax.tick_params(axis="y", labelsize=5.6, pad=1)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.legend(fontsize=4.9, loc="upper left", frameon=False, ncol=1,
              handlelength=1.0, borderpad=0.0, labelspacing=0.2,
              handletextpad=0.4, bbox_to_anchor=(-0.01, 0.99))
    ax.grid(axis="y", linewidth=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("(B)  oracle protocols inflate NDCG@10 by 13–19$\\times$",
                 fontsize=6.6, loc="left", color=INK, pad=2.5)


def panel_c(ax, eta):
    names = [n for n in PANEL_C_ORDER if n in eta]
    vals = [eta[n][0] for n in names]
    wins = [eta[n][1] for n in names]
    ypos = list(range(len(names)))[::-1]

    ax.barh(ypos, vals, height=0.66,
            color=[COLOR[GROUP[n]] for n in names],
            edgecolor=INK, linewidth=0.5, zorder=2)
    for y, v in zip(ypos, vals):
        ax.text(v + 1.8, y, f"{v:.1f}%", va="center", ha="left",
                fontsize=5.2, color=INK)

    ax.axvline(100, color=RED, linestyle=(0, (4, 2.5)), linewidth=1.0, zorder=3)
    ax.text(97.5, len(names) / 2 - 0.4, "oracle reranker  $\\eta = 100\\%$",
            rotation=90, va="center", ha="right", fontsize=5.2, color=RED)
    ax.axvspan(0, 16, color="#F2C94C", alpha=0.25, zorder=0)
    ax.text(18, -1.0, "every method uses $\\leq 16\\%$ of its own ceiling",
            fontsize=5.2, color="#7A5C00", va="center")

    ax.set_xlim(0, 106)
    ax.set_ylim(-1.6, len(names) - 0.3)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", labelsize=5.6, pad=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{n} ($|W|{{=}}{w}$)" for n, w in zip(names, wins)],
                       fontsize=5.2)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel(ETA_XLABEL,
                  fontsize=5.8, labelpad=1.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("(C)  9 LLMs · 12 retrievers · 3 supervised + 1 fine-tuned reranker",
                 fontsize=6.6, loc="left", color=INK, pad=2.5)


def main():
    global ETA_JSON, OUT_DIR, CEILING_TEX, BOUND_LABEL, ETA_XLABEL, ETA_KEY, BOUND_LAST
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["camera-ready", "v2"], default="camera-ready")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.variant == "v2":
        ETA_JSON, ETA_KEY = ETA_V2_JSON, "eta_vs_cor2_own_window"
        OUT_DIR = V2_OUT_DIR
        CEILING_TEX = r"$\mathbb{E}[\mathrm{NDCG}@k]\ \leq\ \mathbb{E}[\mathrm{NDCG}^{*}@k(W_\pi)]$"
        BOUND_LABEL = "closed-candidate bound (Cor. 3)"
        BOUND_LAST = True
        ETA_XLABEL = "ceiling utilisation $\\eta$ = NDCG@10 / NDCG$^*$@10$(W_\\pi)$  (Beauty)"
        for ds in PANEL_B:
            with open(COR2_JSON.format(ds.lower())) as fh:
                PANEL_B[ds]["bound"] = json.load(fh)["corollary2_bound"]["mean"]
    if args.out_dir:
        OUT_DIR = args.out_dir
    eta, raw = load_eta()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "savefig.dpi": 600,
        # ACM requires Type 1 or TrueType fonts only. Matplotlib's PDF default
        # is Type 3, which Sheridan rejects, so force TrueType embedding.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(6.0, 2.42))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.58, 1.30],
                          hspace=0.30, wspace=0.40,
                          left=0.062, right=0.988, top=0.98, bottom=0.115)
    panel_a(fig.add_subplot(gs[0, :]), 2, 19)
    panel_b(fig.add_subplot(gs[1, 0]))
    panel_c(fig.add_subplot(gs[1, 1]), eta)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"fig_overview.{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print("wrote", path)

    lo = min(v[0] for v in eta.values())
    hi = max(v[0] for v in eta.values())
    print(f"eta range across all {len(eta)} methods: {lo:.1f}%-{hi:.1f}% "
          f"(Recall@30={raw['recall_at_30']:.4f}, "
          f"Recall@100={raw['recall_at_100']:.4f})")


if __name__ == "__main__":
    main()
