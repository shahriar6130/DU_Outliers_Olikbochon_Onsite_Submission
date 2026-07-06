# অলীকবচন — Why You're Stuck at 0.55, and the Path to 0.8+

## 0. Folder audit

The folder contains **no code** — only data:

| File | Verdict |
|---|---|
| `train.csv` | Keep. 299 labeled rows (136 hallucinated / 163 faithful). Your only labeled data. |
| `dataset samples.json` | **Redundant** — exact same 299 rows as train.csv in JSON form. Safe to delete. |
| `sample_submission.csv` | Keep (format reference). Note: it's all 1s — submitting it as-is scores F1 = 0.00 on the hallucinated class. |

Your actual notebook lives on Kaggle, so the code-level bottlenecks below are inferred from the data + the score. Drop the notebook into this folder for a line-by-line review.

## 1. The sanity check that should alarm you

Class balance in the sample: ~45.5% hallucinated. If the test set is similar, a **trivial all-zeros submission scores F1(0) ≈ 0.62** (precision .455, recall 1.0). Your 0.55 is *below the constant baseline*. That means your model is biased toward predicting 1 (faithful) and is bleeding recall on class 0 — the class the metric is computed on.

**Do this today:** submit all-zeros once. It calibrates the test-set base rate and probably beats your current score immediately.

## 2. The four real bottlenecks

**B1 — You cannot learn this from 299 rows.** Any BERT/BanglaBERT fine-tune on the sample split overfits surface artifacts (response length, presence of digits) and plateaus at 0.5–0.6. The organizers designed it this way: "the goal is not to fine-tune on our examples." If your pipeline's core is `model.fit(train.csv)`, that is the ceiling you're hitting.

**B2 — ~56% of rows have no context (`[NULL]`).** Any entailment/similarity-against-context approach only works on the 44% that have a passage. On the rest it's a coin flip → blended F1 lands almost exactly where you are: mid-0.5s. Detection requires *world knowledge*, not text matching.

**B3 — One model, six different tasks.** The set mixes: contexted extractive QA, closed-book Bangladesh facts (C1 band — where the competition's weight is), math word problems (~10%, need actual computation), grammar/spelling MCQs, idiom meanings (ভাবার্থ), English vocab. No single classifier handles all of these.

**B4 — Context overlap is a trap, not a feature.** Examples from the sample split:
- Row: "পৃথিবীর দীর্ঘতম হিমবাহ?" → response copies "প্রায় ৫৮ কিলোমিটার" verbatim from context → **label 0** (context says Baltoro is *one of* the longest; the question asks for the longest).
- Row: "কত সালে মুক্তি পায়?" → response "রানওয়ে" (a film name, present in context, but not a year) → **label 0**.

High lexical overlap with context ≠ faithful. The label checks whether the response *answers the question asked* and is *factually right*. Overlap features actively mislead.

## 3. The 0.8+ pipeline

The winning shape is an **LLM-judge router**, not a trained classifier.

### 3.1 Route each item
```
if math/numeric problem  (digits/Bengali numerals + গুণ, সমষ্টি, শতকরা, সম্ভাবনা…) → MATH
elif context present and != [NULL]/empty                                          → GROUNDED
else                                                                              → CLOSED-BOOK
```

### 3.2 GROUNDED (~44%) — cheapest wins first
LLM judge prompt (in Bengali, few-shot with 4–6 sample-split examples) asking three things explicitly:
1. Does the response answer the *type* of question asked (year vs name vs number)?
2. Is it supported by the context?
3. Does it contradict well-known facts even if it echoes the context?

Optionally ensemble with `mDeBERTa-v3-base-xnli` entailment (premise = context+question, hypothesis = response) as a second vote. Expect 0.85–0.92 F1 on this slice.

### 3.3 CLOSED-BOOK (~56%) — this is where your score lives
1. **RAG:** attach a Bengali Wikipedia dump as a Kaggle dataset. BM25 (or `multilingual-e5-small` embeddings) → retrieve top 3–5 passages for the prompt.
2. **Judge with evidence:** same LLM, prompt = question + candidate answer + retrieved passages → verdict.
3. **No/weak evidence found → predict 0.** Abstention should default to hallucinated: the metric is F1 on class 0, and unsupported exotic claims are mostly wrong.

### 3.4 MATH (~10%)
Have the LLM solve step-by-step (or emit Python, execute it) and compare its answer to the candidate. Never ask "is this correct?" directly — solve, then match. The sample math items are mostly hallucinated; solving catches them.

### 3.5 Judge model (fits the rules: ≤50 GB disk, ≤9 h on T4×2/P100)
- **Qwen2.5-14B-Instruct AWQ** — best multilingual quality/VRAM ratio, runs on T4×2 via vLLM. First choice.
- Gemma-2-27B / Qwen2.5-32B GPTQ if runtime budget allows (test 9 h limit on full test size!).
- Llama-3.1-8B as fast fallback / ensemble member.
- **Self-consistency:** sample each verdict 3–5× at T≈0.7, majority vote. Cheap +2–4 F1.

### 3.6 Use the 299 rows correctly
Not for fine-tuning — for **calibration and error analysis**:
- Tune the final decision threshold to maximize F1(class 0) via cross-validation.
- Fit at most a tiny logistic regression over component scores (judge prob, NLI score, retrieval support, route flag). Features generalize; fine-tuned text weights don't.
- Track per-slice F1 (grounded / closed-book / math / MCQ / idiom) so you know which branch to fix.
- Tie-breaker is F1 on the C1 (Bangladesh-specific) subset — retrieval over Bengali Wikipedia/Banglapedia is exactly what helps there.

### 3.7 Rule-compliance notes
- No external APIs — everything above is open-weight and offline. ✔
- Phase 1 predictions may be generated on any hardware you own/rent, **as long as** the same pipeline reproduces inside a Kaggle notebook for Phase 2. Build the Kaggle-runnable version from day one; a leaderboard score you can't reproduce forfeits your finalist slot.
- 4 subs/day: spend them on (1) all-zeros calibration, (2) grounded-branch only, (3) full router — so each submission tells you which slice moved.
- Some sample labels look noisy (e.g., "কম্পিটেন্ট অর্থ অযোগ্য" labeled faithful). Rules require reporting suspected label errors **publicly on the Discussion tab** — doing so is allowed and encouraged.

## 4. Expected arithmetic to 0.8+

| Slice | Share | Achievable F1(0) |
|---|---|---|
| Grounded | ~44% | 0.85–0.92 (judge + NLI) |
| Closed-book factoid | ~46% | 0.75–0.85 (RAG + judge) |
| Math | ~10% | 0.80+ (solve-and-compare) |

Blended: **0.80–0.86** — vs. a hard ceiling of ~0.6 for any approach that ignores the no-context half.

## 5. Priority order (highest ROI first)

1. Submit all-zeros baseline (calibration, likely instant improvement).
2. Stand up Qwen2.5-14B judge, zero-shot, on ALL items → expect ~0.70 already.
3. Add few-shot examples + Bengali prompt + self-consistency → ~0.73–0.76.
4. Add routing + math solve-and-compare → +2–3.
5. Add Bengali Wikipedia RAG for closed-book → +4–6. **This is the 0.8 unlock.**
6. Threshold calibration + small ensemble on the 299 rows → final polish.
