# ============================ JUDGE SCORING (logprob + CoT/self-consistency) ============================
YES_SET = {'yes', 'y', 'true'}
NO_SET  = {'no', 'n', 'false'}

def _to_chat(user_msg):
    msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user_msg}]
    return judge_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def _fit(txt):
    ids = judge_tok.encode(txt)
    if len(ids) > JUDGE_MAXLEN - 8:
        txt = judge_tok.decode(ids[:JUDGE_MAXLEN - 8])
    return txt

def logprob_pfaithful(user_msgs):
    """P(faithful) from the first generated token's Yes/No logprobs."""
    prompts = [_fit(_to_chat(u)) for u in user_msgs]
    if JUDGE_BACKEND == 'vllm':
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outs = llm.generate(prompts, sp)
        res = []
        for o in outs:
            py = pn = 0.0
            lp = o.outputs[0].logprobs
            if lp:
                for tid, l in lp[0].items():
                    s = (l.decoded_token if l.decoded_token is not None
                         else judge_tok.decode([tid])).strip().lower()
                    if s in YES_SET: py += math.exp(l.logprob)
                    elif s in NO_SET: pn += math.exp(l.logprob)
            res.append(py / (py + pn) if (py + pn) > 1e-9 else 0.5)
        return res
    # bnb fallback: read last-position logits
    res = []; B = 8
    yes_ids = set(sum([judge_tok.encode(w, add_special_tokens=False)[:1] for w in ['Yes', 'yes', ' Yes']], []))
    no_ids  = set(sum([judge_tok.encode(w, add_special_tokens=False)[:1] for w in ['No', 'no', ' No']], []))
    for s0 in range(0, len(prompts), B):
        enc = judge_tok(prompts[s0:s0 + B], return_tensors='pt', padding=True,
                        truncation=True, max_length=JUDGE_MAXLEN).to(bnb_model.device)
        with torch.no_grad():
            logits = bnb_model(**enc).logits
        last = enc['attention_mask'].sum(1) - 1
        for bi in range(logits.shape[0]):
            probs = torch.softmax(logits[bi, last[bi]].float(), dim=-1)
            py = float(sum(probs[i] for i in yes_ids)); pn = float(sum(probs[i] for i in no_ids))
            res.append(py / (py + pn) if (py + pn) > 1e-9 else 0.5)
    return res

def cot_pfaithful(user_msgs, n=SC_SAMPLES):
    """P(faithful) by CoT reasoning + majority vote over n samples."""
    if JUDGE_BACKEND != 'vllm':
        return logprob_pfaithful(user_msgs)   # SC needs sampling; bnb -> logprob path
    prompts = [_fit(_to_chat(u)) for u in user_msgs]
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=220, n=n)
    outs = llm.generate(prompts, sp)
    return [self_consistency_pfaithful([c.text for c in o.outputs]) for o in outs]

def _chunked(fn, msgs, **kw):
    out = []
    for s0 in range(0, len(msgs), JUDGE_CHUNK):
        out += fn(msgs[s0:s0 + JUDGE_CHUNK], **kw)
        print(f'    {min(s0 + JUDGE_CHUNK, len(msgs))}/{len(msgs)}')
    return out

def judge_frame(frame, tag):
    """Returns dict index -> P(faithful). Context rows use logprob; closed-book
    rows use CoT+self-consistency (or logprob if SC disabled)."""
    t0 = time.time()
    scores = {}
    if not ENABLE_JUDGE:
        return {i: 0.5 for i in frame.index}
    ctx_idx  = [i for i in frame.index if frame.at[i, 'has_ctx']]
    null_idx = [i for i in frame.index if not frame.at[i, 'has_ctx']]

    print(f'[{tag}] context rows: {len(ctx_idx)} (logprob)')
    if ctx_idx:
        msgs = [prompt_logprob(frame.at[i, 'context'], frame.at[i, 'prompt_bn'],
                               frame.at[i, 'response_bn']) for i in ctx_idx]
        for i, s in zip(ctx_idx, _chunked(logprob_pfaithful, msgs)):
            scores[i] = s

    print(f'[{tag}] closed-book rows: {len(null_idx)} '
          f'({"CoT x%d" % SC_SAMPLES if ENABLE_SELFCONSISTENCY else "logprob"})')
    if null_idx:
        if ENABLE_SELFCONSISTENCY:
            msgs = [prompt_cot('', frame.at[i, 'prompt_bn'], frame.at[i, 'response_bn'])
                    for i in null_idx]
            fn = cot_pfaithful
        else:
            msgs = [prompt_logprob('', frame.at[i, 'prompt_bn'], frame.at[i, 'response_bn'])
                    for i in null_idx]
            fn = logprob_pfaithful
        for i, s in zip(null_idx, _chunked(fn, msgs)):
            scores[i] = s

    print(f'[{tag}] judged in {(time.time() - t0) / 60:.1f} min')
    return scores

judge_train = judge_frame(df, 'train')
judge_test  = judge_frame(test_df, 'test')
df['judge'] = [judge_train[i] for i in df.index]
test_df['judge'] = [judge_test[i] for i in test_df.index]

# quick judge-alone sanity on the labelled set, per regime
if ENABLE_JUDGE:
    for g, sub in df.groupby('has_ctx'):
        pr = [1 if s >= 0.5 else 0 for s in sub['judge']]
        print(f'  has_ctx={g}: judge-alone macroF1={macro_f1(sub["label"].tolist(), pr):.4f} (n={len(sub)})')
