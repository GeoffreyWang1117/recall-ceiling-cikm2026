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

# Check every shipped per-user checkpoint against the published aggregates (~10 s)
python scripts/verify_released_per_user.py
```

The datasets under `data/processed/` are already preprocessed, so no download
step is required to reproduce anything in the paper. `SEED=42` throughout. The
`cikm_*` and `rebuttal_*` drivers are deterministic; the two `kdd_*` drivers
behind Table 3 and §6.1 are not, because their CF-SVD is unseeded (§5(d)).

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
| §6.2 model scaling (7 LLMs) | `scripts/rebuttal_multiapi_n500.py`, `scripts/aggregate_v4_results.py` | `rebuttal_multiapi_n500.json` (restored original, §5(h)); V4 rows of `v4_aggregate_n500.json` only (§5(i)) |
| §6.3 DeepSeek-V4 reasoning models | `scripts/aggregate_v4_results.py` | `v4_aggregate_n500.json` (Movies; Beauty no-think); `rebuttal_multiapi_v4_testrun_20260511.json` (Beauty thinking, n=200) |
| §6.4 leak diagnosis (retracted result) | `scripts/cikm_diagnose_lambdamart_leak.py` | `cikm_lambdamart_leak_diagnostic.json` |
| §6.4 LoRA fine-tuned reranker | `scripts/cikm_finetuned_llm_reranker.py` | `cikm_finetune_llm_n500.json`, `cikm_finetune_llm_n500_electronics.json` |
| §6.5 LLM+CF fusion (RRF, convex) | `scripts/cikm_score_aware_prompting.py` | `cikm_score_aware_prompting.json` |
| §6.6 sequential models | `scripts/kdd_sequential_baselines_all.py` | `kdd_sequential_baselines_all.json` |
| §6.7 controlled GT-injection sweep | `scripts/cikm_controlled_recall_sweep.py` | `cikm_controlled_recall_sweep.json` |
| §6.8 hybrid retrieval (BM25/Dense/RRF) | `scripts/cikm_hybrid_retrieval_v4.py`, `scripts/recompute_hybrid_pvalues.py` | `cikm_hybrid_retrieval_v4.json` (p-values recovered, §5(e)) |
| §6.9 learned cascade fusion | `scripts/cikm_learned_fusion_v4.py` | `cikm_learned_fusion_v4.json` |
| §6.10 closed-catalog robustness | `scripts/cikm_closed_catalog_robustness.py` | `cikm_closed_catalog_v6.json` |
| §6.11 multi-positive last-5 | `scripts/cikm_multipositive_eval_v4.py` | `cikm_multipositive_last5_v4.json` (published); `cikm_multipositive_last5_v4_replication_20260917.json` (replication with per-user arrays and p-values, §5(j)) |
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

## 5. Methods details not stated in the paper

Recorded here after a provenance audit of Table 3 (2026-09-11). None of these
changes the direction of the reported results; all of them affect how the
numbers should be read or reproduced.

**(a) The three primary Amazon test splits are multi-positive, not leave-one-out.**
Beauty, Movies and Electronics carry more than one held-out item per user: mean
items/user is 1.771 on Beauty (62.0% have exactly one), 18.818 on Movies (0.2%),
and 10.062 on Electronics (0.1%). The other five datasets do hold out exactly
one item per user: Office, Sports, Toys and MIND News in `test.parquet`, and
MovieLens-25M in `test_loo.parquet`, which the MovieLens drivers read when it
exists (its `test.parquet` holds five items per user). The driver selects users with `len(test_users[u]) > 0`, not `== 1`, and
scores against the user's full test list. The metric implementations are the
correct multi-positive forms (`ndcg_at_k` uses `IDCG = sum over min(k, |Y|)`,
`recall_at_k` returns `|C ∩ Y| / |Y|`).

Consequence for the bound: the paper states the ceiling through Corollary 2, the
leave-one-out specialisation `E[NDCG@k] <= Recall@|W|`, and quotes
.0823/.0295/.0224. Under multi-positive ground truth Recall@K is not in general
an upper bound on NDCG@k. Counterexample with |Y|=4, k=10 and m=2 retrieved
positives: recall = 0.500 while the oracle reranker attains
IDCG_10(2)/IDCG_10(4) = 1.6309/2.5616 = 0.637. The applicable instrument for
these three datasets is Corollary 3; Corollary 2 remains valid for the other
five. Recomputed with seeded CF-SVD (`scripts/cikm_arm_decomposition.py --mode
bound`), the Corollary 3 bounds are .0911 / .0944 / .0460 for Beauty / Movies /
Electronics, 1.11x / 3.20x / 2.06x the quoted Recall@100 values, which reproduce
exactly. Measured realistic NDCG@10 (.0065 / .0067 / .0050) remains far below
the corrected bounds.

The recomputation ships as `experiments/logs/cikm_corollary2_bound_amazon_{beauty,
movies,electronics}_sampled.json`; each file also stores the per-user arrays the
bound averages over, `per_user.m_retrieved` and `per_user.n_positives`. The
filenames and the `corollary2_bound` key date from before the numbering was
checked and are kept for continuity with the scripts: the quantity inside is the
**Corollary 3** bound. `scripts/cikm_eta_cor2.py` divides the
published NDCG@10 by that bound at each method's own candidate window and writes
`experiments/logs/cikm_eta_windows_cor2.json`; it aborts rather than write if the
user sample it reconstructs does not reproduce the published Recall@30 and
Recall@100 to 1e-12. Table 5 of the camera-ready reports the ceiling utilisation
against the Corollary 2 bound; the corrected values will appear in arXiv v2. The
printed paper is not changed by this release.

**(b) The oracle and realistic arms of Table 3 differ in three ways, not one.**

```python
# realistic: CF order preserved
realistic_cands = cf_candidates[:K]
# oracle:
oracle_candidates = list(cf_candidates[:90])   # (ii) drops CF ranks 91-100
for gt in ground_truth:                        # (i) inserts every positive
    if gt not in oracle_candidates: oracle_candidates.append(gt)
oracle_candidates = oracle_candidates[:K]
np.random.shuffle(oracle_candidates)           # (iii) destroys CF order
```

A fourth difference, found 2026-09-11: `ground_truth` is the user's whole test
list, and the insertion loop adds **every** positive not already present,
including positives that also appear in the user's *training history*. Retrieval
masks history items on purpose (repeat purchases), so these are not items the
retriever missed -- they are items it deliberately suppressed, and no realistic
pipeline would surface them. On Beauty this affects 12.5% of users. The oracle
arm therefore re-admits a class of item the realistic arm can never contain.

The shuffle removes the CF prior from the oracle arm only. Since the LLM largely
defers to CF order in this work (Kendall's tau 0.88-1.00, and Table 7 shows that
greater deference improves NDCG), the oracle arm is handicapped relative to the
realistic arm. The reported 92-95% gap is therefore conservative with respect to
this confound. Anyone reusing this arm as an "oracle protocol" baseline should
note that it bundles three interventions.

**(c) LLM call failures are scored as "the model returned the list unchanged",
not as zero, and were not counted.** `call_llm` returns an empty string after
three failed retries. `parse_ranking` then finds no digits, leaves its index
list empty, and its fill loop appends `0..n-1` in order, so it returns the
**identity permutation**. No exception is raised, so the surrounding
`except Exception: ndcg = 0.0` never fires on an API failure. The effect
therefore differs by arm:

| arm | candidate order on failure | what a failure is scored as |
|---|---|---|
| realistic | CF order | exactly the CF baseline |
| oracle | already shuffled | performance under a random order |

Neither is zero. A failure biases the realistic arm toward CF and the oracle arm
toward random-order performance. `n_in == n_out == 0` is the only trace in the
saved output and nothing checks it, so the failure rate for these runs is
unquantified.

**(d) CF-SVD is not reproducible in the `kdd_*` generation of scripts.**
`TruncatedSVD(n_components=n_comp)` is constructed without `random_state`, and
`cf.fit()` runs *before* `np.random.seed(SEED)`. Measured on Beauty: two
unseeded fits share only **67.8%** of their top-100 candidates, and Recall@100
moves between .0854 and .0893; with `random_state=42` two fits agree exactly.
Seventeen `kdd_*` scripts are affected, including the Table 3 driver. **Every
`cikm_*` and `rebuttal_*` script does pass `random_state=42`**, so the issue is
confined to results produced by the older generation, which in the camera-ready
is principally Table 3. Part of what the paper attributes to "independent user
resampling" across experiments is in fact this SVD randomisation.

**(e) The Wilcoxon p-values in `cikm_hybrid_retrieval_v4.json` were all `null`;
they have been recovered.** `safe_wilcoxon` returns a bare float, but the caller
unpacked it as a tuple (`_, p = safe_wilcoxon(...)`), which raised `TypeError`
into a bare `except` that set `p = None` for all 15 fields. The per-user Recall@100
arrays were saved correctly, so `scripts/recompute_hybrid_pvalues.py` recomputes
the p-values from them after checking every stored mean against its array; no
recall value changes. The paper quotes recall figures only. Note that the ~15%
relative gain of RRF{CF, BM25} on Beauty is not significant (p = .369). The same
bug nulled all nine p-values in `cikm_multipositive_last5_v4.json`; both drivers
are fixed.

**(f) Table 3's confidence intervals used 1,000 bootstrap resamples**, not the
10,000 stated in the paper caption. `bootstrap_ci` defaults to `n_bootstrap=1000`
and none of its four call sites overrides it. The 10,000 figure is correct for a
different analysis, the LLM-minus-CF difference bootstrap in
`rebuttal_bootstrap_test.py`. The bootstrap is also not independently seeded: it
inherits the global NumPy state left after the per-user shuffles, so it
reproduces only on a full in-order re-run.

**(g) Per-user arrays for Table 3 were not persisted.** The driver holds
`oracle_ndcg_scores`, `realistic_ndcg_scores`, `cf_baseline_ndcg_scores` and
`recall_scores` in memory but writes only mean and CI endpoints to
`kdd_oracle_realistic_n500.json`. The intervals in that file match the paper and
can be checked against it, but they cannot be independently recomputed, and no
paired oracle-minus-realistic analysis is possible from the released data. Other
experiments in this repository do ship per-user data; see §6.

**(h) `rebuttal_multiapi_n500.json` was overwritten before release and has been
restored.** The camera-ready commit replaced the original n=500 file with a
test-mode partial run (`n_users` 200, `mode` test, 2026-05-11) that lacks the
Beauty and Movies results of the five older LLMs quoted in Table 5 and §6.2. The
file is restored byte-for-byte from commit `b41f210`, and every one of its 25
model x dataset values is reproduced by the shipped checkpoints. The overwriting
run is kept as `rebuttal_multiapi_v4_testrun_20260511.json`: it is the actual
source of §6.3's Beauty thinking results, which rest on 200 users, not 500.

**(i) Do not reuse the older-model rows of `v4_aggregate_n500.json`.** Its V4
rows match the paper. Its rows for `deepseek-chat` and `gpt-4o-mini` on Movies
equal the CF baseline to fifteen digits, the signature of identity-fallback
re-runs, and its Beauty thinking rows are broken (`n` = 1, NDCG 0).

**(j) The §6.11 result reproduces except for one user.** A CPU re-run in a newer
environment matches 90 of 102 published fields exactly. All differences sit in
the Beauty oracle arm that re-sorts injected candidates by raw CF score: hit@10
moves from 36/309 to 35/309 users (NDCG@10 .0361 to .0358). The realistic arm and
the ideal-placement ceiling match, as do all Movies and Electronics fields; the
most likely cause is a near-tie among low CF scores, unverified because the
original environment no longer exists. The published file is unchanged; the
re-run is shipped separately with per-user arrays and p-values. Beauty evaluates
309 eligible users, not 500. The driver's comment describes a tie-break bonus for
injected positives that the code does not implement, and its oracle re-sort does
not mask training-history items, so injected repeat purchases carry unmasked
CF scores.

**(k) Two V4 checkpoints do not reproduce the published aggregate.** Movies
`deepseek-v4-pro-think` was repaired after aggregation (154 corrupt records); it
reproduces the post-repair value (-42.4%, p = .005) rather than the paper's -43%,
p = .004. Movies `deepseek-v4-pro-nothink` gives .00894 against the published
.00912 (paper: -7%, p = .245); the cause is not identified. Both are declared in
`scripts/verify_released_per_user.py`.

## 6. Per-user data and verification

`experiments/logs/checkpoints/` ships 49 per-user checkpoint files, listed with
their SHA-256, record counts and the paper items they support in
`experiments/logs/per_user_manifest.json`. Only files checked against a published
aggregate are included. Follow-up research checkpoints are not.

| Paper item | Files | Per-user record |
|---|---|---|
| §6.2 (and Table 5, Beauty) | `multiapi_{gpt-4o-mini, llama-3.3-70b-versatile, qwen_qwen3-32b, deepseek-chat, kimi-k2-turbo-preview}_{beauty, movies, electronics, mind, movielens}.jsonl` | `user_id, ndcg, n_in, n_out, latency_s` |
| §6.3 | `multiapi_deepseek-v4-*.jsonl` | same |
| Table 7, §6.5 | `scoreaware_{plain, scoreaware}_{beauty, movies, electronics}.jsonl` | `user_id, llm_order, n_in, n_out, latency_s` (a permutation of the 30-item window) |
| Table 8 | `cikm_ksens_K{10,20,50,100,200}_beauty.jsonl` | `user_id, recall, ndcg_cf, ndcg_llm, ...` |
| §7.2 adaptive-K | `cikm_adaptiveK_{beauty, movies, electronics}.jsonl` | `user_id, K_adapt, fk_recall, fk_ndcg_cf, fk_ndcg_llm, ak_recall, ak_ndcg_cf, ak_ndcg_llm, ...` |

`python scripts/verify_released_per_user.py` recomputes each file's mean and
compares it with the published aggregate (tolerance 5e-5): 60 checks reproduce and
the two declared in §5(k) do not. Score-aware records store permutations, so check
them with `python scripts/cikm_score_aware_prompting.py --offline_only --out
/tmp/sa.json`, which makes no API calls and reproduces all 247 metric fields of
`cikm_score_aware_prompting.json`; only `tokens/*` and `cost_usd` differ, being
zero offline.

Reading the records: the last record per user wins, as in the drivers' resume
logic. A call that failed or returned nothing is scored as the identity
permutation (§5(c)); `n_out == 0` marks an empty response. `user_id` is stored as
a string in most files and as an integer in `cikm_adaptiveK_*`; normalise with
`str()` before joining files. No file contains the same user as both types.

Not shipped because they were never persisted: per-user arrays for Table 3
(§5(g)). Result files that already contain per-user arrays: §6.8, §6.9, and the
§6.11 replication.

## 7. Cost of the API experiments

| Experiment | Models | Cost |
|---|---|---|
| 7-model × 5-dataset comparison | GPT-4o-mini, Qwen3-32B, Llama-3.3-70B, DeepSeek-V3, Kimi-K2, … | $1.24 |
| DeepSeek-V4 standard vs. thinking | V4-Pro, V4-Flash | $4.88 |
| Score-aware prompting + fusion | GPT-4o-mini | $0.42 |

All API drivers checkpoint per user, so an interrupted run resumes without
repaying for completed users. Local models (Gemma3-4B, Qwen3-8B) were served
with Ollama and cost nothing.

## 8. Licence

Code is MIT (see `LICENSE`). The redistributed datasets under `data/processed/`
remain under their original terms (Amazon Review Data, MovieLens-25M, MIND);
see the data notice at the bottom of `LICENSE`.

## 9. Citation

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
