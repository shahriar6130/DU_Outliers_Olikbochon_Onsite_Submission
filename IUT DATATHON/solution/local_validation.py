"""
Local validation harness — অলীকবচন hallucination detection.

Purpose:
  1. Test the ROUTING logic (grounded / closed-book / math) on train.csv.
  2. Provide a plug-in point for any judge function, plus a no-ML heuristic
     floor so the harness runs anywhere (no GPU needed).
  3. Threshold calibration: pick per-route decision thresholds that maximize
     F1 on the HALLUCINATED class (label = 0) via cross-validation.

The real judge (Qwen2.5-14B) runs only on Kaggle — see kaggle_inference.ipynb.
This file exists so you can iterate on routing/calibration logic in seconds.

Usage:  python3 local_validation.py [path/to/train.csv]
"""
import re
import sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Routing (identical logic to the Kaggle notebook — keep in sync)
# ----------------------------------------------------------------------------
MATH_KW = re.compile(
    r"শতকরা|সম্ভাবনা|গুণিতক|মৌলিক|ধারাটি|সমষ্টি|যোগফল|বর্গ(?:মূল|ইঞ্চি|ফুট)?|"
    r"ক্ষতি হ|লাভ হ|বয়স|কত ?দিনে|গড়|অনুপাত|ভগ্নাংশ|সুদ|আসল|"
    r"[0-9০-৯]\s*[+\-*/=]|√|x\s*[*+=]|\*\*|%"
)
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def has_context(ctx) -> bool:
    if ctx is None or (isinstance(ctx, float) and np.isnan(ctx)):
        return False
    s = str(ctx).strip()
    return s not in ("", "[NULL]", "NULL", "nan")


def is_math(prompt: str) -> bool:
    return bool(MATH_KW.search(str(prompt)))


def route(row) -> str:
    if is_math(row["prompt_bn"]):
        return "math"
    return "grounded" if has_context(row["context"]) else "closedbook"


# ----------------------------------------------------------------------------
# Heuristic judge (floor only — the LLM judge replaces this on Kaggle)
# Returns a score in [0, 1]: higher = more likely FAITHFUL (label 1).
# ----------------------------------------------------------------------------
YEAR_Q = re.compile(r"কবে|কত সালে|কোন সালে|সাল|তারিখ")
NUM_Q = re.compile(r"কত|কয়|সংখ্যা")
HAS_NUM = re.compile(r"[0-9০-৯]")
STOP = set("কী কি কে কার কাকে কোন কোথায় কবে হয় ছিল ছিলেন এর একটি ও এবং থেকে সালে করে করা হয়েছিল".split())


def _tokens(s: str):
    s = str(s).translate(BN_DIGITS)
    return [t for t in re.findall(r"[\wঀ-৿]+", s) if t not in STOP and len(t) > 1]


def heuristic_score(row) -> float:
    r = route(row)
    resp, q = str(row["response_bn"]), str(row["prompt_bn"])
    if r == "math":
        return 0.4  # weak prior: math candidates are ~50/50, lean hallucinated
    if r == "closedbook":
        return 0.45  # 53% of closed-book sample rows are hallucinated
    # grounded: answer-type check + content overlap with context
    ctx = str(row["context"])
    if YEAR_Q.search(q) and NUM_Q.search(q) and not HAS_NUM.search(resp):
        return 0.05  # asked for a year/number, response has none -> trap
    rt = _tokens(resp)
    if not rt:
        return 0.3
    ov = sum(t in ctx.translate(BN_DIGITS) for t in rt) / len(rt)
    return 0.25 + 0.6 * ov


# ----------------------------------------------------------------------------
# Metric + threshold calibration
# ----------------------------------------------------------------------------
def f1_hallucinated(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 0) & (y_true == 1)).sum())
    fn = int(((y_pred == 1) & (y_true == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def best_threshold(scores, y_true, grid=None):
    """Threshold t: predict 1 iff score >= t. Maximize F1(class 0)."""
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_hallucinated(y_true, (np.asarray(scores) >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def calibrate_per_route(df, scores, n_folds=5, seed=42):
    """Out-of-fold per-route threshold calibration. Returns (thresholds, oof_f1)."""
    rng = np.random.RandomState(seed)
    folds = rng.permutation(len(df)) % n_folds
    routes = df.apply(route, axis=1).values
    scores = np.asarray(scores)
    y = df["label"].values
    oof_pred = np.ones(len(df), dtype=int)
    for k in range(n_folds):
        tr, va = folds != k, folds == k
        for rt in ("grounded", "closedbook", "math"):
            m_tr, m_va = tr & (routes == rt), va & (routes == rt)
            t = best_threshold(scores[m_tr], y[m_tr])[0] if m_tr.sum() >= 10 else 0.5
            oof_pred[m_va] = (scores[m_va] >= t).astype(int)
    final_thresholds = {
        rt: best_threshold(scores[routes == rt], y[routes == rt])[0]
        for rt in ("grounded", "closedbook", "math")
        if (routes == rt).sum() >= 10
    }
    return final_thresholds, f1_hallucinated(y, oof_pred)


# ----------------------------------------------------------------------------
def main(path="../train.csv"):
    df = pd.read_csv(path)
    assert {"context", "prompt_bn", "response_bn", "label"} <= set(df.columns)
    routes = df.apply(route, axis=1)
    print(f"rows: {len(df)}   label split: {df.label.value_counts().to_dict()}")
    print("route counts:", routes.value_counts().to_dict())
    for rt in routes.unique():
        m = routes == rt
        print(f"  {rt:<10} n={m.sum():>3}  hallucinated={int((df.label[m]==0).sum())}")

    # baselines
    print("\n-- baselines (F1 on hallucinated class) --")
    print(f"all-zeros : {f1_hallucinated(df.label, np.zeros(len(df))):.4f}")
    print(f"all-ones  : {f1_hallucinated(df.label, np.ones(len(df))):.4f}")

    scores = df.apply(heuristic_score, axis=1).values
    th, oof = calibrate_per_route(df, scores)
    print(f"\nheuristic judge  OOF F1(0): {oof:.4f}   thresholds: {th}")
    print("(this is the no-ML floor — the Kaggle LLM judge replaces heuristic_score)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../train.csv")
