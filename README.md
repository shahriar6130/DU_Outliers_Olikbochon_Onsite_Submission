# অলীকবচন — Bengali LLM Hallucination Detection

**Team DU_Outliers** — submission for the BrainLab / Institute of Policy Dynamics datathon at IUT 12th ICT
Fest 2026.

## Problem

Given a Bengali prompt and a candidate response — sometimes with a supporting context passage, sometimes
without — predict whether the response is **faithful** (`label=1`) or **hallucinated** (`label=0`).
Scored on binary F1 of the hallucinated class. No conventional training set is provided; a 299-row
labeled sample (`train.csv`) is released for local pipeline validation only, and the real test set is
held out for the Kaggle leaderboard.

## Approach in one paragraph

A single open-weight judge model (Qwen2.5-32B-Instruct-AWQ, run locally via vLLM, fully compliant with
the no-external-API rule) scores every response two ways — a single-token Yes/No log-probability
("lp32") and a chain-of-thought self-consistency vote ("cot32") — and the two signals are blended
per-route with weights and a decision threshold fit on labeled calibration data. Rows are routed into
one of three prompt strategies before scoring: **grounded** (context given, judge checks faithfulness to
the passage), **closedbook** (no context, judge checks factual correctness from its own knowledge plus
BM25-retrieved Bengali Wikipedia evidence), and **math** (regex-detected arithmetic/word-problem
questions, judge solves independently and compares). A deterministic QA-bank lookup layer can resolve
some closedbook rows without a judge call at all when a near-exact matching question is found in an
attached answer bank.

## Pipeline (by notebook cell)

| Cell | Stage |
|---|---|
| 1 | Engine install (vLLM, offline/online dual-mode, CUDA-13 compatibility shims) |
| 2 | Config, input discovery (`find_test_csv`, `find_labeled_sample`, `find_wikis`), route thresholds |
| 3 | Prompt templates, routing logic (`route_row`), math-keyword regex |
| 4 | BM25 index built over every attached Wikipedia/QA-bank corpus |
| 5 | Deterministic QA-bank override (options-verified — see below) |
| 6 | E5 semantic retrieval (`HYBRID` — **on**, content-hash-validated cache, see Configuration flags) |
| 7 | LoRA verifier pass (currently **off** — leakage risk, see Known limitations) |
| 8 | vLLM judge engine load, prompt formatting, judge-response caching |
| 9 | Prompt builders (`p_grounded`, `p_closedbook`, `p_math`) and verdict parsing |
| 10 | Per-row scoring: bank overrides first, then adaptive lp32 + cot32 allocation for whatever's left |
| 11 | Per-route blend-weight + threshold fitting via 5-fold CV (linear blend vs. logistic stacker, CV-selected) |
| 12 | Final blend, thresholding, `submission.csv` + `test_scores.csv` output |
| 13 | Notes |

## Configuration flags (cell 2 / cell 6 / cell 11)

| Flag | Current value | Why |
|---|---|---|
| `USE_LORA` | `False` | The fine-tuned verifier adapter was trained on the same 299-row calibration sample used to fit blend weights — any weight learned from it is leakage. Two real submissions with it enabled scored below the no-LoRA baseline. |
| `HYBRID` | `True` | E5 semantic retrieval (cell 6), re-enabled after fixing the root cause of the prior stale-cache bug: the cache is now validated against a SHA-1 fingerprint of the corpus content, not just its row count, so a reordered corpus can no longer silently poison a cached embedding file. |
| `RAG_MAX_PASSAGES` | `900_000` | Raised from `500_000` after 3 overlapping QA-bank dataset attachments were found to truncate the Wikipedia corpus tail in a real run. |
| `ADAPTIVE_ROUTES` | `True` | Rows where the lp32 pass already reads confidently (`<=0.05` or `>=0.95`) skip the CoT judge call entirely; the calls saved fund `k_base+2` votes on rows genuinely near the decision boundary. Same or less total compute than fixed-K sampling. |
| `K_GROUNDED`, `K_CLOSEDBOOK`, `K_MATH` | `3, 5, 3` (base) | Chain-of-thought votes per row; auto-reduced to `2, 4` when the test set exceeds 4,000 rows, to stay inside the 9-hour runtime cap. |
| Stacker (cell 11) | linear blend or logistic regression, chosen per route by 5-fold CV | A small L2-regularized logistic regression over `[lp32, cot32]` is tried alongside the linear weighted blend on every fold; whichever wins on held-out performance for that route is used in the final fit — never selected from in-sample fit alone. |
| `BANK_DOCS` matching | options-verified | A bank match is only trusted when the test response appears among the matched question's listed options — this excludes template-question collisions (different question, same wording shape) that previously caused a real regression. |

## External data (declared, public, non-test-derived)

| Source | Role | License / citation |
|---|---|---|
| `wikimedia/wikipedia`, config `20231101.bn` | Bengali Wikipedia passage corpus (grounded/closedbook evidence retrieval) | CC BY-SA — built once via `prep_build_wiki_index.ipynb`, republished as a public Kaggle dataset per the External Data Policy |
| shazol Bangla Wikipedia corpus (Kaggle dataset) | Additional Wikipedia passage source, merged into the same BM25 index | Public Kaggle dataset |
| BCS exam question bank (10th–45th BCS + Bangladesh Bank exam, Bangla language/literature) | QA-bank retrieval + deterministic bank-override matching | Public exam archive (scribd.com mirrors), declared on the competition Discussion tab |
| `hishab/titulm-bangla-mmlu` | QA-bank retrieval corpus | Public Hugging Face dataset |
| `sakhadib/bagdhara-bangla-idioms-dataset` | QA-bank retrieval corpus (idiom Q&A entries, tagged `বাগধারা:` in the bank format) | Public Kaggle dataset |
| BanglaHalluEval (organizer-provided benchmark, `BanglaHalluEval-EB77`) | Extended calibration data — scored through the same judge and blended into per-route weight fitting only, never into the honest cross-validation number, and never touching the competition test set | Provided directly by the competition organizers for this purpose |

All of the above are public, cited, and were not derived from the competition's test set, per the rules'
External Data Policy. Full external dataset / model / vLLM wheel links are in
[`Dataset/links.md`](Dataset/links.md), including the BanglaHalluEval extended-calibration set
(`bangla-hallu-eval-1200`, `benhallueval-v1`), the cached corpus embeddings (`corpus-emb-v13`), and the
judge response cache (`judge-cache`).

## Models used

| Model | Role |
|---|---|
| Qwen2.5-32B-Instruct-AWQ | Primary judge — lp32 (single-token logprob) and cot32 (chain-of-thought vote) signals |
| Qwen2.5-14B-Instruct-AWQ | Fallback judge, only used if the 32B snapshot fails to load (`JUDGE_LADDER`) |
| Qwen2.5-14B-Instruct + LoRA adapter | Third verifier signal — implemented, currently disabled (`USE_LORA=False`, see Known limitations) |
| `multilingual-e5-small` | Semantic retrieval encoder for hybrid BM25+embedding evidence — enabled (`HYBRID=True`), content-hash cache validation |

All models are open-weight, loaded from Kaggle-attached datasets/snapshots, run entirely locally via
vLLM — no external API calls, per the competition's compute rules.

## Compute budget

Fits within Kaggle's code-competition limits: GPU T4×2, vLLM AWQ quantization, no fine-tuning at
inference time. Estimated runtime 3–5 hours for a ~2,500-row test set, well under the 9-hour cap;
`K_GROUNDED`/`K_CLOSEDBOOK` auto-reduce for larger held-out folds (organizer notice: up to ~5,000 rows).
On-disk model size (32B AWQ + 14B AWQ fallback) is under the 50GB cap.

## How to run

**Phase 1 (leaderboard, internet ON is fine):** attach competition data, Wikipedia corpora, one QA-bank
dataset, the 32B AWQ snapshot. Run all cells. `submission.csv` is written to the working directory.

**Phase 2 (offline reproduction):** additionally attach the vLLM wheels dataset. The notebook
auto-detects a local model snapshot and enforces `HF_HUB_OFFLINE=1` — no code changes needed between
Phase 1 and Phase 2 runs. Do not attach more than one QA/bank-named dataset at once (cell 2 prints a
warning if it detects this — each overlapping attachment eats into the passage cap and can silently
truncate the Wikipedia corpus).

## Known limitations (honest, not for show)

- **Structural CV/leaderboard gap.** Every real submission across this project has scored below its own
  cross-validation number. Root cause: the 299-row labeled sample is small relative to the real test set
  and doesn't fully represent its difficulty distribution — confirmed via a prior regression where 378
  real test rows were affected by a failure mode invisible in the 299-row sample. The extended-calibration
  blend (BanglaHalluEval-derived data) is the direct countermeasure, but it does not close this gap by
  construction — it narrows it with more representative labeled data.
- **`USE_LORA` is implemented but disabled.** The adapter was fine-tuned on the same 299 rows used for
  calibration, so any weight learned from its output is leakage, not signal. It would need to be
  retrained on fully held-out data (e.g., the BanglaHalluEval calibration extension) to be usable safely.
- **Closedbook is the least-protected route.** With `lp32`/`cot32` as the only signals and no bank match,
  it has zero fallback if the judge's own world knowledge is wrong — this is also the route where the
  competition's C1 (Bangladesh-specific) cultural-distance band concentrates, per the problem statement.
- **`HYBRID` retrieval was re-enabled this session** after fixing its cache-validation bug, but has not
  yet been validated against a real Kaggle run with the actual judge — only against synthetic data to
  confirm the code path executes correctly. Set `HYBRID=False` in cell 6 for a lower-risk run if needed.

## Repository contents

- `IUT DATATHON/DU_Outliers_inference_notebook.ipynb` — the submission notebook described above.
- `prep_build_wiki_index.ipynb` — one-time prep notebook, builds the Bengali Wikipedia passage corpus.
- `Dataset/links.md` — every external dataset, model, and vLLM wheel link used by the inference notebook.
