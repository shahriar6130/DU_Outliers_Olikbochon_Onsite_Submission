# ============================ PROMPT DESIGN ============================
def _clip(s, n):
    s = s or ''
    return s if len(s) <= n else s[:n] + '…'

# few-shot exemplars: shortest 2-per-class closed-book rows from the labelled set
_nc = df[~df.has_ctx].copy()
FEWSHOT_IDX = []
if len(_nc) >= 4:
    _nc['tl'] = _nc.prompt_bn.str.len() + _nc.response_bn.str.len()
    FEWSHOT_IDX = (list(_nc[_nc.label == 1].sort_values('tl').index[:2]) +
                   list(_nc[_nc.label == 0].sort_values('tl').index[:2]))
print('few-shot exemplar rows (excluded from calibration):', FEWSHOT_IDX)

def fewshot_block():
    lines = []
    for i in FEWSHOT_IDX:
        r = df.loc[i]
        v = 'Yes' if int(r.label) == 1 else 'No'
        lines.append(f'Question: {_clip(r.prompt_bn, 200)}\n'
                     f'Proposed answer: {_clip(r.response_bn, 200)}\nVerdict: {v}')
    return '\n\n'.join(lines)

FEWSHOT = fewshot_block()
SYS = ('You are a meticulous bilingual (Bengali/English) fact-checker. '
       'You judge whether a Bengali answer is factually correct and, when a '
       'passage is given, fully supported by it.')

def prompt_logprob(ctx, q, resp):
    """Single-token Yes/No prompt — used for CONTEXT rows (fast, NLI-like)."""
    ctx, q, resp = _clip(ctx, CTX_CLIP), _clip(q, PROMPT_CLIP), _clip(resp, RESP_CLIP)
    if has_context(ctx):
        return (f'Passage (Bengali):\n{ctx}\n\nQuestion (Bengali): {q}\n'
                f'Response (Bengali): {resp}\n\n'
                'Is the response factually correct AND fully supported by the passage, '
                'with no fabricated or contradicting detail? '
                'Answer with exactly one word: Yes or No.')
    head = ('Here are solved examples:\n\n' + FEWSHOT + '\n\n') if FEWSHOT else ''
    return (head + f'Now judge this one.\nQuestion (Bengali): {q}\n'
            f'Proposed answer (Bengali): {resp}\n'
            'Is the proposed answer factually correct? '
            'Answer with exactly one word: Yes or No.\nVerdict:')

def prompt_cot(ctx, q, resp):
    """Reason-then-verdict prompt — used for CLOSED-BOOK (and RAG) rows."""
    ctx, q, resp = _clip(ctx, CTX_CLIP), _clip(q, PROMPT_CLIP), _clip(resp, RESP_CLIP)
    if has_context(ctx):
        return (f'Passage (Bengali):\n{ctx}\n\nQuestion (Bengali): {q}\n'
                f'Response (Bengali): {resp}\n\n'
                'Reason in 1-3 short sentences about whether the response is factually '
                'correct and supported by the passage. Then, on a new line, output exactly '
                '"Verdict: Yes" if it is faithful, or "Verdict: No" if it is hallucinated.')
    head = ('Solved examples:\n\n' + FEWSHOT + '\n\n') if FEWSHOT else ''
    return (head + 'Now judge the following using your own knowledge of Bengali/Bangladesh '
            'facts. Reason in 1-3 short sentences, then on a new line output exactly '
            '"Verdict: Yes" if the answer is factually correct, or "Verdict: No" if it is '
            f'wrong or fabricated.\nQuestion (Bengali): {q}\n'
            f'Proposed answer (Bengali): {resp}')
