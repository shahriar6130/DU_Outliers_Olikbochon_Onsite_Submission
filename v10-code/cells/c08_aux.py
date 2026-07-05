# ============================ OPTIONAL AUX FEATURES (low-weight tie-break) ============================
# Default OFF (W_JUDGE=1.0). Turn on ENABLE_AUX and set W_JUDGE<1.0 only if the
# judge-vs-blend comparison in the next cell shows the blend wins on the dev set.
aux_train = [0.5] * len(df)
aux_test  = [0.5] * len(test_df)

if ENABLE_AUX:
    try:
        # free the judge to make room for the small encoders
        if ENABLE_JUDGE and JUDGE_BACKEND == 'vllm':
            try:
                from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
                del llm; destroy_model_parallel(); destroy_distributed_environment()
            except Exception:
                pass
        gc.collect()
        if HAS_CUDA: torch.cuda.empty_cache()

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline as SKPipe
        dev = 'cuda' if HAS_CUDA else 'cpu'

        # NLI entailment / contradiction
        ntok = AutoTokenizer.from_pretrained(NLI_MODEL)
        nmod = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL, torch_dtype=torch.float16 if HAS_CUDA else torch.float32).to(dev).eval()
        def nli_feats(fr):
            ent, con = [], []
            for s0 in range(0, len(fr), 32):
                sub = fr.iloc[s0:s0 + 32]
                enc = ntok(list(sub['premise']), list(sub['response_bn']), truncation=True,
                           max_length=512, padding=True, return_tensors='pt').to(dev)
                with torch.no_grad():
                    p = torch.softmax(nmod(**enc).logits.float(), -1).cpu().numpy()
                ent += list(p[:, 0]); con += list(p[:, 2])
            return np.array(ent), np.array(con)
        e_tr, c_tr = nli_feats(df); e_te, c_te = nli_feats(test_df)
        del nmod; gc.collect()
        if HAS_CUDA: torch.cuda.empty_cache()

        # e5 similarity
        em = SentenceTransformer(EMB_MODEL, device=dev)
        def sim(fr):
            r = em.encode(['query: ' + t for t in fr['response_bn']], normalize_embeddings=True)
            p = em.encode(['passage: ' + t for t in fr['premise']], normalize_embeddings=True)
            return np.sum(r * p, axis=1)
        s_tr, s_te = sim(df), sim(test_df)
        del em; gc.collect()
        if HAS_CUDA: torch.cuda.empty_cache()

        def hand(fr):
            out = []
            for _, r in fr.iterrows():
                rt = set(re.findall(r'[ঀ-৿a-zA-Z0-9]+', r['response_bn']))
                pt = set(re.findall(r'[ঀ-৿a-zA-Z0-9]+', r['premise']))
                jac = len(rt & pt) / max(len(rt | pt), 1)
                out.append([jac, len(rt - pt) / max(len(rt), 1), float(len(rt) <= 1), float(r['has_ctx'])])
            return np.array(out)
        h_tr, h_te = hand(df), hand(test_df)

        X_tr = np.column_stack([df['judge'].values, e_tr, c_tr, s_tr, h_tr])
        X_te = np.column_stack([test_df['judge'].values, e_te, c_te, s_te, h_te])
        lr = SKPipe([('sc', StandardScaler()),
                     ('lr', LogisticRegression(C=0.3, max_iter=2000, class_weight='balanced'))])
        lr.fit(X_tr, y)
        aux_train = lr.predict_proba(X_tr)[:, 1].tolist()
        aux_test  = lr.predict_proba(X_te)[:, 1].tolist()
        print('aux features + LR ready')
    except Exception as e:
        print('aux skipped:', type(e).__name__, str(e)[:200])
else:
    print('aux disabled (judge-only).')
