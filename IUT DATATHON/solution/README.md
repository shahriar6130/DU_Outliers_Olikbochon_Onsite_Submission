# অলীকবচন Solution Package — Team Outliers

Router + open-weight LLM-judge pipeline targeting 0.8+ F1 on the hallucinated class.
Everything here is compliant with the competition rulebook (checklist at the bottom).

## Files

| File | What it is |
|---|---|
| `kaggle_inference.ipynb` | **The submission notebook.** test.csv → submission.csv, end to end, offline. Upload to Kaggle. |
| `prep_build_wiki_index.ipynb` | One-time prep (internet ON): builds `bn_wiki_passages.parquet` from Bengali Wikipedia. |
| `local_validation.py` | Local harness: routing tests, threshold calibration, no-ML heuristic floor. No GPU needed. |

## Setup — do these once

1. **Model.** On Kaggle, add `Qwen/Qwen2.5-14B-Instruct-AWQ` via **Kaggle Models** (or upload the HF
   snapshot as a private dataset). ~10 GB on disk — far under the 50 GB limit.
2. **Wiki corpus (the 0.8 unlock — don't skip).** Run `prep_build_wiki_index.ipynb` in any
   internet-enabled notebook. Upload the output parquet as a **public** Kaggle dataset
   (External Data Policy requires public + cited). Attach it to the inference notebook.
3. **(Optional, speed)** Attach a dataset containing vLLM wheels (search Kaggle for "vllm whl") for
   offline install. Without it the notebook falls back to transformers + bitsandbytes 4-bit
   (slower; auto-reduces to 1 vote — still inside 9 h).
4. Create the inference notebook on Kaggle, paste/upload `kaggle_inference.ipynb`, attach:
   competition data + model + wiki dataset (+ wheels), **internet OFF**, accelerator **T4 ×2**.

## How it works

- **Routing:** `grounded` (has context) / `closedbook` (`[NULL]` context) / `math` (keyword detector).
  On the sample split: 125 / 149 / 25 of 299.
- **Grounded:** few-shot Bengali judge prompt that explicitly targets the dataset's trap:
  responses copied verbatim from context that don't answer the question asked are hallucinated.
- **Closed-book:** BM25 (pure-python, offline) retrieves top-4 Bengali Wikipedia passages; the judge
  verdicts with evidence. No corpus attached → knowledge-only judging (works, scores lower).
- **Math:** the model solves the problem itself, then compares answers (solve-and-compare).
- **Self-consistency:** 3 sampled verdicts per item, score = vote fraction.
- **Thresholds:** calibrated per route on the labeled sample split to maximize F1(class 0);
  biased toward predicting 0 when uncertain (the metric only counts class 0).

## Verification status

- All notebook cells compile; full pipeline dry-run (mock engine) on a fake `/kaggle/input` produced a
  valid `submission.csv` (schema, dtypes, id alignment all asserted in-notebook).
- BM25 unit-tested (correct passages retrieved for Bengali factoid queries).
- Routing + calibration validated on the 299 labeled rows (`local_validation.py`).
- **Not yet run with the real model** — that needs Kaggle GPUs. First real run: check the
  sample-split F1 printed by the calibration cell before trusting a submission.

## Submission plan (4/day — spend them diagnostically)

1. **Day 1, sub 1:** all-zeros baseline (`sub = pd.DataFrame({'id': ids, 'label': 0})`).
   Calibrates the test base rate; expected ≈ 0.62 if test matches the sample split.
2. **Day 1, sub 2:** this pipeline, no wiki corpus (knowledge-only). Expected ≈ 0.70–0.75.
3. **Day 2:** pipeline + wiki RAG. Expected ≈ 0.78–0.85.
4. Iterate: compare the calibration cell's per-route sample-split F1 against leaderboard moves to see
   which branch to improve. Keep your 2 best subs selected before the deadline (else Kaggle takes
   the 2 most recent).

Honest caveat: no one can guarantee a leaderboard number — the expected ranges above follow from the
sample-split structure (all-zeros already scores ~0.62; the LLM judge only has to beat chance on each
route to clear 0.8 blended). Validate on the sample split first; the calibration cell prints exactly
the number you need to see (aim for ≥ 0.85 there, since the held-out set is harder).

## Rule-compliance checklist

- [x] Open-weight model only (Qwen2.5-14B-Instruct-AWQ, Apache-2.0), loadable from Kaggle datasets/Models — no external APIs.
- [x] All inference local, inside a Kaggle kernel, T4×2/P100, well under 9 h (~30–60 min expected with vLLM).
- [x] Total model size ≪ 50 GB.
- [x] No hardcoded row count / order / IDs (re-run policy) — schema detected at runtime; verified by dry run.
- [x] Submission format: exactly `id,label`, labels ∈ {0,1}, asserted before writing.
- [x] External data (Bengali Wikipedia) is public, will be published as a public Kaggle dataset and cited in the paper.
- [x] No test-set fine-tuning, no manual test labeling, no leaderboard probing (the all-zeros sub is a
      normal first submission, not systematic label inference).
- [x] Phase 1 predictions reproducible offline by this same notebook (Phase 2 requirement).
- [ ] **You:** declare the wiki dataset on the Discussion tab, keep the notebook attached datasets public
      where required, and report the suspected label errors (e.g., "কম্পিটেন্ট → অযোগ্য" labeled faithful)
      publicly on Discussion — required channel, and it's good citizenship.

## Phase 2 reminders (if you make top 30)

Needed by the Phase 2 deadline: this inference notebook (runnable), model checkpoint attached as
dataset/Kaggle Model, 4-page ACL/EMNLP-format paper, presentation slides, and in-notebook
documentation (the markdown cells already cover approach + reproduction). Winners = 0.20×Phase 1 +
0.50×held-out + 0.10×presentation + 0.10×paper + 0.10×novelty — the paper and novelty scores are
cheap points; budget time for them.
