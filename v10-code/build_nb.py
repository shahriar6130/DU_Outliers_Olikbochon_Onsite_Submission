# -*- coding: utf-8 -*-
"""Assemble model_v10_0.ipynb from validated cell files + markdown."""
import json, os

CELLDIR = 'cells'
def code(fname):
    return open(os.path.join(CELLDIR, fname), encoding='utf-8').read()

INSTALL = (
"# ============================ SETUP ============================\n"
"# vLLM pins torch/transformers -> install it FIRST, everything else without -U.\n"
"!pip install -q vllm\n"
"!pip install -q sentence-transformers faiss-cpu peft accelerate bitsandbytes\n"
"!pip install -q git+https://github.com/csebuetnlp/normalizer\n"
)

MD_TITLE = """# Bangla Hallucination Detection — v10 "Judge + RAG"

**Pipeline:** two-regime routing → LLM judge (context rows = single-token Yes/No
logprob; closed-book rows = **chain-of-thought + self-consistency, k=%d**) →
**RAG** over an attached Bengali-Wikipedia corpus for closed-book rows →
optional low-weight NLI/e5/hand blend → **one regularized threshold per regime**
calibrated on the labelled set → `submission.csv`. No pseudo-labeling.

**Why:** ~46%% of the test set is `[NULL]` context (closed-book). Consistency-only
models (BanglaBERT/NLI/e5) are near-random there and cap out ~0.65. World
knowledge comes from the local open LLM (APIs are banned) plus retrieval.

### Attach on Kaggle (Add Data)
1. **Competition data** — labelled samples (`dataset samples.json`/`train.csv`) + `test set.csv`.
2. **Synthetic dev (optional)** — `synthetic_train_5000.csv` (this repo's `synthetic-data/`).
3. **Bengali Wikipedia (for RAG)** — any public parquet/csv/json with a Bengali text
   column (e.g. a `wikimedia/wikipedia` bn export, or a Bangla passages corpus).
   RAG **auto-skips** if none is attached, so the notebook still runs.

Internet must be OFF at submission (re-run policy). All models are open-weight;
declare model + license + this synthetic data per the rulebook.
""" % (5,)

MD_CORE = "## Core logic (routing, verdict parsing, self-consistency, metrics, calibration)\nUnit-tested on CPU — see `test_core.py` in the repo (25/25 passed)."
MD_DATA = "## 1. Load & normalize · route into context vs closed-book"
MD_PROMPT = "## 2. Prompts — logprob (context) and CoT (closed-book / RAG)"
MD_ENGINE = "## 3. Judge engine — vLLM ladder (Qwen2.5-32B→14B→7B AWG) with bnb 4-bit fallback"
MD_JUDGE = "## 4. Run the judge — logprob on context rows, CoT+self-consistency on closed-book rows"
MD_RAG = "## 5. RAG — retrieve Bengali-Wikipedia evidence and re-judge closed-book rows (auto-skips if no corpus)"
MD_AUX = "## 6. Optional aux features (NLI + e5 + hand + LR) — low-weight tie-break, OFF by default"
MD_FINAL = "## 7. Calibrate per-regime thresholds, compare judge-vs-blend, write submission"
MD_SUMMARY = """## Summary (for the report / 7-min video)

Judge (Qwen2.5-AWQ via vLLM) with **two regimes** — single-token logprob for
grounded rows, **CoT + self-consistency** for closed-book rows — then **RAG**
over Bengali Wikipedia to inject world knowledge into the 46%% closed-book slice,
then **one regularized threshold per regime**. Pseudo-labeling and the heavy
ML stack were dropped (they overfit the tiny labelled set / hurt private LB).
Novelty: regime-aware judging + retrieval-grounded closed-book fact-checking for
low-resource Bengali. Fine-tuning path in `train_lora_v10.ipynb`.
"""

def c(cell_type, src):
    return {"cell_type": cell_type, "metadata": {},
            **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
            "source": src.splitlines(keepends=True)}

cells = [
    c("markdown", MD_TITLE),
    c("code", INSTALL),
    c("code", code('c01_config.py')),
    c("markdown", MD_CORE),
    c("code", code('c02_core.py')),
    c("markdown", MD_DATA),
    c("code", code('c03_data.py')),
    c("markdown", MD_PROMPT),
    c("code", code('c04_prompts.py')),
    c("markdown", MD_ENGINE),
    c("code", code('c05_engine.py')),
    c("markdown", MD_JUDGE),
    c("code", code('c06_judge.py')),
    c("markdown", MD_RAG),
    c("code", code('c07_rag.py')),
    c("markdown", MD_AUX),
    c("code", code('c08_aux.py')),
    c("markdown", MD_FINAL),
    c("code", code('c09_finalize.py')),
    c("markdown", MD_SUMMARY),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

out = 'model_v10_0.ipynb'
json.dump(nb, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', out, '-', len(cells), 'cells')
