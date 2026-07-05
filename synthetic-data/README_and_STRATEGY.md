# Synthetic dataset + model/training strategy

## 1. The 5,000-row dataset (`synthetic_train_5000.csv` / `.json`)

Same schema as the competition: `context, prompt_bn, response_bn, label`
(`1` = faithful, `0` = hallucinated). Fully reproducible from `make_dataset.py`
(fixed seed, no network, no external data — so it is compliant with the rulebook
and citable as "competition-runtime generated data").

**Composition (this build):**

| Segment | Rows | Label correctness |
|---|---|---|
| Context-grounded (intrinsic) | ~3,100 (62%) | **Guaranteed** — the passage *is* the ground truth |
| Computable math / logic (closed-book) | ~1,050 | **Guaranteed** — answer verified in code |
| Curated closed-book facts (idioms, EN→BN word meaning, spelling, general knowledge) | ~460 (with paraphrase + distractor variants) | High-confidence hand-curated |
| Balance | 2,500 faithful / 2,500 hallucinated | — |

Hallucination types mirror the real data: number/date swaps, entity swaps,
unsupported/fabricated additions, cross-slot (answer to the wrong question),
antonym/wrong-meaning, and wrong spelling.

**Honest limitation.** The trustworthy core is the context-grounded half
(which directly matches the ~46% of the *test* set that has context) and the
computable math. Real closed-book *world facts* cannot be safely synthesised
without a real corpus. So use this set for **(a) a validation harness,
(b) a few-shot pool, (c) light fine-tuning of the output format** — and for the
closed-book world-knowledge slice, *add a real public corpus on Kaggle*
(`csebuetnlp/squad_bn`, Bengali Wikipedia). Those are public + declarable and
fully legal under the External Data Policy.

To regenerate or resize: edit the `emit_*` targets in `make_dataset.py` and run
`python make_dataset.py`.

---

## 2. Why you are stuck at ~0.65 (recap)

46% of the test is `[NULL]` context = closed-book fact-checking. Your
BanglaBERT / NLI / embedding / overlap features only measure *response↔context
consistency*; on closed-book rows there is nothing to be consistent with, so
they are near-random there. That caps any consistency-only model near 0.65.
Breaking the ceiling requires **world knowledge**, which — because external APIs
are banned — must come from a **local open-source LLM** and/or **retrieval**.

---

## 3. Model choice

APIs are banned; the open model *is* your ceiling on closed-book Bengali facts,
so pick it empirically. Benchmark these three on the real 299 rows (and this
synthetic dev set), reading F1 **separately for context vs closed-book**:

- **Gemma-2-27B-it** — strongest Bengali world-knowledge in the open weights that fits Kaggle (AWQ/4-bit on 2×T4 or L4×4).
- **Qwen2.5-14B-Instruct** — best reasoning-per-VRAM; fast; your v9 already wires it up.
- **Aya-Expanse-32B** (Cohere) — explicitly multilingual, often best on Bangla-specific facts.

**Recommended default:** run the judge as **Gemma-2-27B-it (primary) + Qwen2.5-14B (secondary)** and average their P(faithful). If compute is tight, ship **Qwen2.5-14B alone** — it is the efficiency-prize-friendly choice. Whichever wins the *closed-book* split of the 299 wins the competition; do not choose on overall F1.

---

## 4. Training / inference strategy (the recipe to reach 0.8+)

Ranked by impact. Steps 1–3 are the jump from 0.65→~0.75; step 4 (RAG) is what
pushes past 0.80; steps 5–6 protect the private LB (70% of the online score).

1. **Two-regime routing.** Context rows → grounded faithfulness check. `[NULL]`
   rows → knowledge check. Never send them through the same prompt/threshold.

2. **Chain-of-thought + self-consistency on closed-book rows.** Replace the
   single-token Yes/No with: let the model reason 2–3 sentences, then emit a
   verdict; sample k=5 and majority-vote. Keep the fast single-token logprob
   path only for context rows (there it behaves like NLI and is fine). This is
   the single biggest lever for the closed-book half.

3. **Light LoRA (QLoRA, 4-bit) to specialise the verifier.** Fine-tune the base
   LLM to output a clean `Yes/No` (and short rationale) on **real 299 (weighted
   high) + this 5,000 synthetic set (format/robustness)**. Fine-tuning is
   explicitly allowed and is your novelty for the paper. Keep it light so you
   teach the *task format*, not synthetic artifacts — world knowledge stays in
   the frozen weights + RAG.

4. **RAG for `[NULL]` rows (external data is legal).** Attach an offline Bengali
   Wikipedia dump as a Kaggle dataset; retrieve top-k passages per closed-book
   question with your e5 embedder; feed them to the judge as evidence. This
   converts closed-book → grounded and directly attacks the weak 46%. Highest
   novelty — good for the 30% offline (presentation + paper) score.

5. **One regularised threshold per regime.** Drop the 30×30 two-threshold grid +
   nested candidate selection — on 299 rows that overfits public LB. Pick one
   threshold per regime by simple CV on the 299 (+synthetic), tuned to the
   *actual* metric (confirm on Kaggle whether it's macro-F1 or F1-on-class-0).

6. **Drop pseudo-labeling.** It historically hurt you (v3 → 0.466) and adds
   variance for no reliable gain on a strong judge.

7. **Prefer judge-dominant over the ML blend.** On a good judge, `judge_only`
   with the regime thresholds usually beats the LR/XGB blend on the *private*
   set. Keep NLI/e5/hand features only as a tie-breaker, low-weight.

**Deployment decider:** judge (CoT+self-consistency, ±LoRA) → RAG evidence on
`[NULL]` → per-regime threshold → submit. Optional 2-model average.

---

## 5. Compliance checklist (rulebook)

- No external API — all inference local (open weights via vLLM). ✔
- Open models: disclose name/version/repo + license + claimed pretraining data. ✔
- External data (this synthetic set, squad_bn, BN Wikipedia): public + declared,
  not derived from the test set. ✔
- Inference notebook must run within Kaggle limits and be reproducible. ✔
- No manual labelling of test data. ✔
