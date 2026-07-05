# -*- coding: utf-8 -*-
"""
v10 core logic — pure python, no GPU / no heavy deps.
Everything here is unit-tested on CPU so the non-LLM parts of the pipeline are
guaranteed correct. The inference notebook embeds these exact functions.
"""
import re, math

# ---------------------------------------------------------------- #
#  text / routing
# ---------------------------------------------------------------- #
NO_CTX = {'', 'nan', 'naN', 'NaN', '[null]', '[NULL]', 'null', 'NULL', 'none', 'None'}

def has_context(ctx):
    if ctx is None:
        return False
    return str(ctx).strip().lower() not in {s.lower() for s in NO_CTX}

def clean_text(t, normalize=None):
    if t is None:
        return ''
    s = str(t).strip()
    if s.lower() in {x.lower() for x in NO_CTX}:
        return ''
    if normalize is not None:
        try:
            return normalize(s)
        except Exception:
            return s
    return s

# ---------------------------------------------------------------- #
#  verdict parsing for CoT + self-consistency
# ---------------------------------------------------------------- #
YES_WORDS = ['yes', 'faithful', 'correct', 'supported', 'true',
             'হ্যাঁ', 'হাঁ', 'সঠিক', 'সত্য', 'সমর্থিত']
NO_WORDS = ['no', 'hallucinat', 'incorrect', 'unsupported', 'false', 'wrong', 'fabricat',
            'না', 'ভুল', 'মিথ্যা', 'অসমর্থিত', 'ভিত্তিহীন']

_VERDICT_RE = re.compile(
    r'(?:verdict|final answer|answer|উত্তর|রায়|সিদ্ধান্ত)\s*[:：\-]?\s*'
    r'(yes|no|হ্যাঁ|না|সঠিক|ভুল|correct|incorrect|faithful|hallucinated)',
    re.IGNORECASE)

def parse_verdict(text):
    """Return 1 (faithful/Yes), 0 (hallucinated/No), or None if unclear."""
    if not text:
        return None
    t = text.strip().lower()
    # 1) explicit "Verdict: X" near the end wins
    matches = _VERDICT_RE.findall(text)
    if matches:
        tok = matches[-1].lower()
        if tok in ('yes', 'সঠিক', 'correct', 'faithful', 'হ্যাঁ'):
            return 1
        if tok in ('no', 'ভুল', 'incorrect', 'hallucinated', 'না'):
            return 0
    # 2) last standalone yes/no token
    last = None
    for m in re.finditer(r'\b(yes|no)\b|(হ্যাঁ|না|সঠিক|ভুল)', t):
        last = m.group(0)
    if last is not None:
        if last in ('yes', 'হ্যাঁ', 'সঠিক'):
            return 1
        if last in ('no', 'না', 'ভুল'):
            return 0
    # 3) fall back to keyword tally
    y = sum(t.count(w) for w in YES_WORDS)
    n = sum(t.count(w) for w in NO_WORDS)
    if y > n:
        return 1
    if n > y:
        return 0
    return None

def self_consistency_pfaithful(samples, fallback=0.5):
    """samples: list of generated strings. Return P(faithful) by majority vote,
    ignoring unparseable samples; fallback if none parse."""
    votes = [parse_verdict(s) for s in samples]
    votes = [v for v in votes if v is not None]
    if not votes:
        return fallback
    return sum(votes) / len(votes)

# ---------------------------------------------------------------- #
#  metrics (pure python; matches sklearn f1_score average='macro' / binary)
# ---------------------------------------------------------------- #
def _f1(y, p, pos):
    tp = fp = fn = 0
    for yi, pi in zip(y, p):
        if pi == pos and yi == pos:
            tp += 1
        elif pi == pos and yi != pos:
            fp += 1
        elif pi != pos and yi == pos:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def macro_f1(y, p):
    return 0.5 * (_f1(y, p, 0) + _f1(y, p, 1))

def score_metric(y, p, metric='macro'):
    if metric == 'macro':
        return macro_f1(y, p)
    if metric == 'hallucinated':      # F1 of the hallucinated class (label 0)
        return _f1(y, p, 0)
    if metric == 'faithful':
        return _f1(y, p, 1)
    raise ValueError(metric)

# ---------------------------------------------------------------- #
#  regularized threshold calibration
# ---------------------------------------------------------------- #
def calibrate_threshold(scores, labels, metric='macro',
                        grid=None, prefer=0.5):
    """Single threshold. Coarse grid, tie-break toward `prefer` (=0.5) to avoid
    overfitting a tiny dev set. pred = 1 if score >= t."""
    if grid is None:
        grid = [round(0.20 + 0.025 * i, 3) for i in range(25)]   # 0.20..0.80
    best_t, best_s = prefer, -1.0
    for t in grid:
        pred = [1 if s >= t else 0 for s in scores]
        sc = score_metric(labels, pred, metric)
        if sc > best_s + 1e-9 or (abs(sc - best_s) <= 1e-9 and abs(t - prefer) < abs(best_t - prefer)):
            best_t, best_s = t, sc
    return best_t, best_s

def calibrate_two_regime(scores, labels, is_ctx, metric='macro'):
    """One threshold for context rows, one for closed-book rows — each calibrated
    only on its own subset, regularized toward 0.5."""
    sc_c = [s for s, g in zip(scores, is_ctx) if g]
    lb_c = [l for l, g in zip(labels, is_ctx) if g]
    sc_n = [s for s, g in zip(scores, is_ctx) if not g]
    lb_n = [l for l, g in zip(labels, is_ctx) if not g]
    t_ctx = calibrate_threshold(sc_c, lb_c, metric)[0] if sc_c else 0.5
    t_null = calibrate_threshold(sc_n, lb_n, metric)[0] if sc_n else 0.5
    return t_ctx, t_null

def route_predict(scores, is_ctx, t_ctx, t_null):
    return [1 if (s >= (t_ctx if g else t_null)) else 0
            for s, g in zip(scores, is_ctx)]

# ---------------------------------------------------------------- #
#  blend (judge-dominant)
# ---------------------------------------------------------------- #
def blend(judge, aux=None, w_judge=0.8):
    """Judge-dominant blend. aux is an optional secondary probability list."""
    if aux is None:
        return list(judge)
    return [w_judge * j + (1 - w_judge) * a for j, a in zip(judge, aux)]
