# অলীকবচন — Bengali LLM Hallucination Detection

**Team Outliers** — submission for the BrainLab / Institute of Policy Dynamics datathon at IUT 12th ICT
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
| 6 | E5 semantic retrieval (`HYBRID` — currently **off**, see Known limitations) |
| 7 | LoRA verifier pass (currently **off** — leakage risk, see Known limitations) |
| 8 | vLLM judge engine load, prompt formatting, judge-response caching |
| 9 | Prompt builders (`p_grounded`, `p_closedbook`, `p_math`) and verdict parsing |
| 10 | Per-row scoring: bank overrides first, then lp32 + cot32 for whatever's left |
| 11 | Per-route blend-weight + threshold fitting via 5-fold CV on the labeled sample |
| 12 | Final blend, thresholding, `submission.csv` + `test_scores.csv` output |
| 13 | Notes |

## Configuration flags (cell 2 / cell 6 / cell 2)

| Flag | Current value | Why |
|---|---|---|
| `USE_LORA` | `False` | The fine-tuned verifier adapter was trained on the same 299-row calibration sample used to fit blend weights — any weight learned from it is leakage. Two real submissions with it enabled scored below the no-LoRA baseline. |
| `HYBRID` | `False` | E5 semantic retrieval is implemented (cell 6) but disabled — a corpus-reorder previously went undetected by a row-count-only cache check. Safe to re-enable once the cache validation is content-based, not just row-count-based. |
| `RAG_MAX_PASSAGES` | `900_000` | Raised from `500_000` after 3 overlapping QA-bank dataset attachments were found to truncate the Wikipedia corpus tail in a real run. |
| `K_GROUNDED`, `K_CLOSEDBOOK`, `K_MATH` | `3, 5, 3` | Chain-of-thought votes per row; auto-reduced to `2, 4` when the test set exceeds 4,000 rows, to stay inside the 9-hour runtime cap. |
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
External Data Policy. The BanglaHalluEval-derived extended-calibration CSVs and the notebook that scores
them (`score_ext_calib.ipynb`) were built during this project but are not currently present in this local
folder — if you need to regenerate them, see **Known gaps** below.

## Models used

| Model | Role |
|---|---|
| Qwen2.5-32B-Instruct-AWQ | Primary judge — lp32 (single-token logprob) and cot32 (chain-of-thought vote) signals |
| Qwen2.5-14B-Instruct-AWQ | Fallback judge, only used if the 32B snapshot fails to load (`JUDGE_LADDER`) |
| Qwen2.5-14B-Instruct + LoRA adapter | Third verifier signal — implemented, currently disabled (`USE_LORA=False`, see Known limitations) |
| `multilingual-e5-small` | Semantic retrieval encoder for hybrid BM25+embedding evidence — implemented, currently disabled (`HYBRID=False`) |

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
  blend (BanglaHalluEval-derived data, when attached) is the direct countermeasure, but it does not close
  this gap by construction — it narrows it with more representative labeled data.
- **`HYBRID` (E5 semantic retrieval) is implemented but disabled.** It was turned off after a stale
  cached embedding file went undetected by a cache check that only compared row counts, not content —
  re-enabling it safely requires a content-based cache validation (e.g., a corpus fingerprint) before
  trusting a cached embedding file across runs.
- **`USE_LORA` is implemented but disabled.** The adapter was fine-tuned on the same 299 rows used for
  calibration, so any weight learned from its output is leakage, not signal. It would need to be
  retrained on fully held-out data (e.g., the BanglaHalluEval calibration extension) to be usable safely.
- **Closedbook is the least-protected route.** With `lp32`/`cot32` as the only signals and no bank match,
  it has zero fallback if the judge's own world knowledge is wrong — this is also the route where the
  competition's C1 (Bangladesh-specific) cultural-distance band concentrates, per the problem statement.

## Known gaps in this folder (vs. what's been built during this project)

This folder currently holds `kaggle_inference_v16.ipynb` and `prep_build_wiki_index.ipynb`. Several
supporting artifacts referenced in the notebook's own logic (it looks for `calib_scores_ext.csv` and
`banglahallueval_ext_calib_*.csv` under `/kaggle/input`) were built earlier in this project but are not
present locally right now:

- `score_ext_calib.ipynb` — one-time notebook that scores an extended calibration sample through the
  same judge.
- `banglahallueval_ext_calib_*.csv` — the extended calibration sample itself (source-grouped,
  class-balanced subsamples of the BanglaHalluEval benchmark).
- `prep_build_qa_bank.ipynb` and `bangla_bcs_qs.parquet` — build the QA-bank retrieval corpus.

If these were already uploaded as Kaggle datasets, this is not a problem — the inference notebook reads
them from `/kaggle/input` regardless of what's mirrored in this local folder. If they need to be rebuilt,
say so and they can be regenerated.

**One more thing worth flagging directly: this exact notebook file does not currently include three
engineering changes (adaptive compute allocation, a properly cache-validated hybrid retrieval fix, and a
cross-validation-selected logistic-regression stacker) that were built and validated in this project's
most recent working session.** If the file in this folder was restored from an older Kaggle notebook
version, those changes aren't in it. Say the word and they can be reapplied.
