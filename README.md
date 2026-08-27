# The Recall Ceiling of LLM Recommendation Reranking

Code, processed datasets, and raw result files for the CIKM 2026 paper:

> Zhaohui Wang. **The Recall Ceiling of LLM Recommendation Reranking.**
> In *Proceedings of the 35th ACM International Conference on Information and
> Knowledge Management (CIKM '26)*, Rome, Italy, November 7–11, 2026.

**Project page — <https://geoffreywang1117.github.io/recall-ceiling-cikm2026/>**
The theorem, the 92–95% oracle gap, an interactive RAEP diagnostic that takes
your own Recall@|W| and returns the ceiling it puts on your NDCG, and the full
ledger of seventeen strategies that failed to beat a $0.00 CF baseline.

Every number in the paper comes from a JSON file in `experiments/logs/`, and
every one of those files is produced by a script in `scripts/`. The table below
maps each paper float to the script that produces it and the result file it
reads. Nothing in the paper is hand-entered without a corresponding file here.

---

## 1. What is in this repository

```
scripts/            experiment drivers (one per experiment; see the mapping below)
evaluation/         metric implementations (NDCG@k, Hit@k, MAP@k, Recall@K)
models/             retrieval and reranker implementations
utils/              data loading, prompt construction, LLM API clients
analysis/           statistical tests (Wilcoxon, Holm, bootstrap)
data/processed/     preprocessed datasets (3-core filtered, chronological LOO split)
experiments/logs/   every result JSON referenced by the paper
```

## 2. Quickstart

```bash
conda create -n recallceiling python=3.11 && conda activate recallceiling
pip install -r requirements.txt

# API-based experiments only: copy and fill in your own keys
cp .env.example .env

# Reproduce the ceiling-utilisation numbers (no API calls, ~30 s)
python scripts/cikm_recompute_eta_windows.py

# Reproduce Figure 1 from the result JSONs
python scripts/make_fig_overview.py
```

The datasets under `data/processed/` are already preprocessed, so no download
step is required to reproduce anything in the paper. `SEED=42` throughout; all
non-API experiments are deterministic.

## 3. Paper float → script → result file

| Paper float | Script | Result file |
|---|---|---|
| Fig. 1 (overview) | `scripts/make_fig_overview.py` | `cikm_eta_windows.json` |
| Tab. 1 (protocol audit) | — (literature audit; see §2 of the paper) | — |
| Tab. 2 (dataset statistics) | `scripts/cikm_dataset_stats.py` (recomputes and checks the table against the shipped splits) | `cikm_dataset_stats.json` |
| Tab. 3 (oracle vs. realistic) | `scripts/kdd_oracle_realistic_n500.py` | `kdd_oracle_realistic_n500.json` |
| Tab. 4 (retrieval sweep) | `scripts/kdd_sota_retrieval_baselines.py`, `scripts/cikm_lightgcn_sweep.py` | `kdd_sota_retrieval_baselines.json`, `cikm_lightgcn_sweep.json` |
| Tab. 5 (all rerankers on Beauty) | `scripts/cikm_recompute_eta_windows.py` | `cikm_eta_windows.json` |
| Tab. 6 (supervised rerankers, 6 datasets) | `scripts/cikm_neural_rerankers_disjoint.py` | `cikm_neural_rerankers_disjoint.json` |
| Tab. 7 (score-aware prompting) | `scripts/cikm_score_aware_prompting.py` | `cikm_score_aware_prompting.json` |
| Tab. 8 (candidate-budget sweep) | `scripts/cikm_k_sensitivity_n500.py` | `cikm_k_sensitivity_n500.json` |

### Results reported in prose

| Paper section | Script | Result file |
|---|---|---|
| §5.3 multi-metric consistency | `scripts/rebuttal_multi_metric.py` | `rebuttal_multi_metric.json` |
| §5.4 cross-domain (MovieLens, MIND) | `scripts/kdd_movielens_experiments.py`, `scripts/llm_reranking_new_datasets.py` | `kdd_movielens_experiments.json`, `llm_reranking_new_datasets.json` |
| §5.5 density analysis | `scripts/density_recall_analysis.py` | `density_recall_analysis.json` |
| §6.1 prompt engineering | `scripts/kdd_oracle_vs_realistic_prompts.py` | `kdd_oracle_vs_realistic_prompts.json` |
| §6.2 model scaling (7 LLMs) | `scripts/rebuttal_multiapi_n500.py`, `scripts/aggregate_v4_results.py` | `rebuttal_multiapi_n500.json`, `v4_aggregate_n500.json` |
| §6.3 DeepSeek-V4 reasoning models | `scripts/aggregate_v4_results.py` | `v4_aggregate_n500.json` |
| §6.4 leak diagnosis (retracted result) | `scripts/cikm_diagnose_lambdamart_leak.py` | `cikm_lambdamart_leak_diagnostic.json` |
| §6.4 LoRA fine-tuned reranker | `scripts/cikm_finetuned_llm_reranker.py` | `cikm_finetune_llm_n500.json`, `cikm_finetune_llm_n500_electronics.json` |
| §6.5 LLM+CF fusion (RRF, convex) | `scripts/cikm_score_aware_prompting.py` | `cikm_score_aware_prompting.json` |
| §6.6 sequential models | `scripts/kdd_sequential_baselines_all.py` | `kdd_sequential_baselines_all.json` |
| §6.7 controlled GT-injection sweep | `scripts/cikm_controlled_recall_sweep.py` | `cikm_controlled_recall_sweep.json` |
| §6.8 hybrid retrieval (BM25/Dense/RRF) | `scripts/cikm_hybrid_retrieval_v4.py` | `cikm_hybrid_retrieval_v4.json` |
| §6.9 learned cascade fusion | `scripts/cikm_learned_fusion_v4.py` | `cikm_learned_fusion_v4.json` |
| §6.10 closed-catalog robustness | `scripts/cikm_closed_catalog_robustness.py` | `cikm_closed_catalog_v6.json` |
| §6.11 multi-positive last-5 | `scripts/cikm_multipositive_eval_v4.py` | `cikm_multipositive_last5_v4.json` |
| §6.12 statistical power | `scripts/kdd_statistical_power_analysis.py` | `kdd_statistical_power_analysis.json` |
| §7.1 cost audit | — (provider price lists) | `cikm_cost_table.json` |
| §7.2 adaptive-$K$ | `scripts/cikm_adaptive_k_ndcg.py` | `cikm_adaptive_k_ndcg.json` |
| §7.2 RAEP audit of LLMRank | `scripts/cikm_raep_audit_llmrank.py` | `cikm_raep_audit_llmrank.json` |

`experiments/logs/` also contains result files from earlier versions of this
work (the `kdd_*` and `p9_*` families). They are kept so that the paper's
self-corrections can be checked against the original runs, but the CIKM paper
cites only the files listed above.

## 4. Three things you should know before reading the results

**(a) The listwise LLM prompts see 30 candidates, not 100.** Retrieval always
returns `K=100`, but the listwise prompt shows the top-30 CF candidates and
appends positions 31–100 below the model's output in CF order
(`rebuttal_multiapi_n500.py`, `aggregate_v4_results.py`). The pointwise
rerankers — LambdaMART, RankNet, MLP, and the LoRA-fine-tuned LLM — score all
100 (`cikm_neural_rerankers_disjoint.py`, `cikm_finetuned_llm_reranker.py`).
The ceiling-utilisation ratio η is therefore computed against each method's own
window, not a common `K`; `cikm_recompute_eta_windows.py` does this and prints
both denominators. Scoring a 30-window LLM against Recall@100 understates its η
by ≈2.6×.

**(b) Amazon Electronics has no item text.** All 29,905 items in the sampled
subset carry `title: null` and `text: null`. Every LLM run on Electronics
therefore saw opaque ASINs and no semantic content, which is why its
plain-prompt Kendall's τ against the CF order is exactly 1.000. Electronics
remains a valid test for retrieval and for the CF-embedding rerankers, but its
LLM results are a null control, not a measurement of semantic reranking.

**(c) One earlier result in this line of work was retracted.** An earlier
version reported `p < 0.0001` gains for LambdaMART above 8.8% recall. Those
runs had overlapping training and evaluation user pools, so each user's test
item was simultaneously a supervision label and an evaluation target.
`cikm_diagnose_lambdamart_leak.py` reproduces the leak and shows it disappears
under any non-leaky split; `cikm_neural_rerankers_disjoint.py` is the corrected
protocol (250 training users disjoint from the 500 evaluation users). The
diagnostic is kept in this repository on purpose.

## 5. Cost of the API experiments

| Experiment | Models | Cost |
|---|---|---|
| 7-model × 5-dataset comparison | GPT-4o-mini, Qwen3-32B, Llama-3.3-70B, DeepSeek-V3, Kimi-K2, … | $1.24 |
| DeepSeek-V4 standard vs. thinking | V4-Pro, V4-Flash | $4.88 |
| Score-aware prompting + fusion | GPT-4o-mini | $0.42 |

All API drivers checkpoint per user, so an interrupted run resumes without
repaying for completed users. Local models (Gemma3-4B, Qwen3-8B) were served
with Ollama and cost nothing.

## 6. Licence

Code is MIT (see `LICENSE`). The redistributed datasets under `data/processed/`
remain under their original terms (Amazon Review Data, MovieLens-25M, MIND);
see the data notice at the bottom of `LICENSE`.

## 7. Citation

```bibtex
@inproceedings{wang2026recallceiling,
  author    = {Wang, Zhaohui},
  title     = {The Recall Ceiling of {LLM} Recommendation Reranking},
  booktitle = {Proceedings of the 35th ACM International Conference on
               Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  address   = {Rome, Italy},
  publisher = {ACM},
  doi       = {10.1145/3799682.3841132}
}
```
