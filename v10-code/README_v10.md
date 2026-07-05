# v10 "Judge + RAG" — run guide

Implements the full strategy: two-regime routing → LLM judge (logprob on context
rows, **CoT + self-consistency** on closed-book rows) → **RAG** over Bengali
Wikipedia → optional low-weight blend → **one regularized threshold per regime**.
No pseudo-labeling, no heavy ML stack (both overfit the 299-row labelled set).

## Files
- `myNotebooks/model_v10_0.ipynb` — **the submission notebook** (self-contained, Kaggle-runnable).
- `myNotebooks/train_lora_v10.ipynb` — optional QLoRA verifier training (produces an adapter).
- `v10-code/hallu_v10_core.py` — pure logic (routing, verdict parsing, self-consistency, metrics, calibration).
- `v10-code/test_core.py` — CPU unit tests (25/25 pass) + a mock end-to-end on the real 299.
- `v10-code/cells/`, `v10-code/build_nb.py` — the notebook is assembled from these validated cells (edit + rebuild).

## What was verified vs what runs on Kaggle
- **Verified on CPU here:** all routing / self-consistency / metric / threshold /
  submission logic (`test_core.py`, 25/25), and a full non-GPU dry-run over the
  real `dataset samples.json` + `test set.csv` that produced a correctly-formatted
  `submission.csv` (2,516 rows, `id,label`).
- **Runs only on Kaggle (GPU, no APIs):** the vLLM judge, CoT+self-consistency,
  and RAG. This code follows the proven vLLM pattern from your v9 and has graceful
  fallbacks at every step, but the *score* must be produced by running on Kaggle.

## Kaggle setup (Add Data)
1. **Competition data** — labelled samples (`dataset samples.json` or `train.csv`) + `test set.csv`.
2. **Synthetic dev (optional)** — `synthetic_train_5000.csv` from `synthetic-data/`.
3. **Bengali Wikipedia (for RAG)** — any public parquet/csv/json with a Bengali
   text column (e.g. a `wikimedia/wikipedia` bn export, or a Bangla passages
   corpus). RAG **auto-detects** it and **auto-skips** if absent — the notebook
   still runs without it.

## Run order
1. (Optional) Run `train_lora_v10.ipynb` on a 24GB+ GPU → host the adapter →
   set `USE_LORA_ADAPTER=True`, `LORA_PATH=...` in the inference notebook.
2. Run `model_v10_0.ipynb` end-to-end (GPU, internet OFF). It writes
   `/kaggle/working/submission.csv`.

## Key toggles (config cell)
| Toggle | Default | Note |
|---|---|---|
| `ENABLE_SELFCONSISTENCY`, `SC_SAMPLES` | True, 5 | CoT majority vote on closed-book rows |
| `ENABLE_RAG` | True | auto-skips if no wiki corpus attached |
| `JUDGE_LADDER` | Qwen2.5-32B→14B→7B AWQ | swap in `google/gemma-2-27b-it` / `CohereForAI/aya-expanse-32b` to compare |
| `METRIC` | `'macro'` | set to `'hallucinated'` if Kaggle's metric is F1 on class 0 |
| `ENABLE_AUX`, `W_JUDGE` | False, 1.0 | judge-only by default; enable blend only if it wins the dev readout |

## Tuning to 0.8+ (do these in order, checking the labelled-set readout each time)
1. **Pick the model.** Try Qwen2.5-32B, gemma-2-27b-it, aya-expanse-32b; keep the
   one with the best **closed-book** (`has_ctx=False`) F1 — that regime is the ceiling.
2. **Attach a good Bengali Wikipedia corpus** and confirm `RAG=on` in the final print.
3. **Confirm the metric** (`macro` vs `hallucinated`) and re-calibrate.
4. Only then consider `ENABLE_AUX=True` / LoRA if the honest readout improves.

## Compliance (rulebook)
Open-weight models only, all inference local (no APIs). Declare model +
version/repo + license, and declare the synthetic data (`make_dataset.py` is
fully reproducible). RAG corpus must be public + declared, not derived from the
test set. Inference notebook is reproducible within Kaggle limits.
