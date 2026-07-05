# ============================ RAG over Bengali Wikipedia (closed-book rows) ============================
# Attach a public Bengali-Wikipedia dataset on Kaggle (any parquet/csv/json with a
# text column, e.g. 'wikimedia/wikipedia' bn split exported to parquet, or
# 'bengali-wikipedia-passages'). This stage self-skips if none is found or anything
# fails — the judge scores from the previous cell remain the fallback.
RAG_OK = False
RAG_BLEND = 0.5          # final_null = (1-b)*closed_book_judge + b*rag_judge

def _find_wiki():
    cand = []
    for root in ['/kaggle/input']:
        for ext in ('*.parquet', '*.csv', '*.json', '*.jsonl'):
            cand += glob.glob(os.path.join(root, '**', ext), recursive=True)
    # prefer files that look like a wiki / passages corpus, and NOT the competition data
    bad = ('test set', 'test_set', 'dataset samples', 'sample submission', 'synthetic_train')
    cand = [c for c in cand if not any(b in c.lower() for b in bad)]
    cand = [c for c in cand if any(k in c.lower() for k in ('wiki', 'passage', 'corpus', 'bangla', 'bengali'))]
    return sorted(cand, key=lambda p: (-os.path.getsize(p), len(p)))

def _load_passages(path):
    txtcols = ('text', 'passage', 'content', 'body', 'context', 'article', 'paragraph')
    if path.endswith('.parquet'):
        d = pd.read_parquet(path)
    elif path.endswith(('.json', '.jsonl')):
        d = pd.read_json(path, lines=path.endswith('.jsonl'))
    else:
        d = pd.read_csv(path)
    col = next((c for c in d.columns if c.lower() in txtcols), None)
    if col is None:
        col = max(d.columns, key=lambda c: d[c].astype(str).str.len().mean())
    passages = []
    for t in d[col].astype(str).tolist():
        t = clean_text(t, normalize)
        if len(t) < 40:
            continue
        for s0 in range(0, len(t), RAG_PASSAGE_CHARS):
            passages.append(t[s0:s0 + RAG_PASSAGE_CHARS])
            if len(passages) >= RAG_MAX_PASSAGES:
                return passages
    return passages

if ENABLE_RAG and ENABLE_JUDGE and JUDGE_BACKEND == 'vllm':
    try:
        wikis = _find_wiki()
        assert wikis, 'no Bengali-Wikipedia dataset attached'
        print('RAG corpus:', wikis[0])
        passages = _load_passages(wikis[0])
        assert len(passages) > 50, 'corpus too small'
        print(f'RAG passages: {len(passages)}')

        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(EMB_MODEL, device='cuda' if HAS_CUDA else 'cpu')
        P = emb.encode(['passage: ' + p for p in passages], batch_size=128,
                       normalize_embeddings=True, show_progress_bar=True).astype('float32')

        try:
            import faiss
            index = faiss.IndexFlatIP(P.shape[1]); index.add(P)
            def _search(qv, k):
                _, I = index.search(qv, k); return I
        except Exception:
            print('faiss unavailable -> numpy top-k')
            def _search(qv, k):
                sims = qv @ P.T
                return np.argpartition(-sims, min(k, sims.shape[1] - 1), axis=1)[:, :k]

        def rag_rejudge(frame, tag):
            idx = [i for i in frame.index if not frame.at[i, 'has_ctx']]
            if not idx:
                return {}
            Q = emb.encode(['query: ' + frame.at[i, 'prompt_bn'] for i in idx],
                           batch_size=128, normalize_embeddings=True,
                           show_progress_bar=False).astype('float32')
            I = _search(Q, RAG_TOPK)
            msgs = []
            for row_n, i in enumerate(idx):
                ev = '\n'.join(passages[j] for j in I[row_n] if 0 <= j < len(passages))
                msgs.append(prompt_cot(ev, frame.at[i, 'prompt_bn'], frame.at[i, 'response_bn']))
            print(f'[RAG {tag}] re-judging {len(idx)} closed-book rows with retrieved evidence')
            sc = _chunked(cot_pfaithful, msgs)
            return dict(zip(idx, sc))

        rag_train = rag_rejudge(df, 'train')
        rag_test  = rag_rejudge(test_df, 'test')
        for frame, rag in [(df, rag_train), (test_df, rag_test)]:
            newv = []
            for i in frame.index:
                if i in rag:
                    newv.append((1 - RAG_BLEND) * frame.at[i, 'judge'] + RAG_BLEND * rag[i])
                else:
                    newv.append(frame.at[i, 'judge'])
            frame['judge'] = newv
        del emb, P; gc.collect()
        if HAS_CUDA: torch.cuda.empty_cache()
        RAG_OK = True
        print('RAG applied to closed-book rows.')
    except Exception as e:
        print('RAG skipped:', type(e).__name__, str(e)[:200])
else:
    print('RAG disabled or judge not on vLLM — skipping.')
