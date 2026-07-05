# ============================ LOAD & NORMALIZE ============================
try:
    from normalizer import normalize
except Exception:
    print('normalizer unavailable -> identity'); normalize = lambda s: s

DATA_PATH = find_file('dataset samples.json', 'dataset_samples.json', 'train.csv')
TEST_PATH = find_file('test set.csv', 'test_set.csv', 'test.csv')
assert DATA_PATH and TEST_PATH, 'attach the competition data (labelled samples + test set)'
print('labelled:', DATA_PATH, '\ntest    :', TEST_PATH)

if DATA_PATH.endswith('.json'):
    df = pd.DataFrame(json.load(open(DATA_PATH, encoding='utf-8')))
else:
    df = pd.read_csv(DATA_PATH)
test_df = pd.read_csv(TEST_PATH)
if 'id' not in test_df.columns:
    test_df.insert(0, 'id', np.arange(1, len(test_df) + 1))

for d in (df, test_df):
    for c in ['context', 'prompt_bn', 'response_bn']:
        d[c] = d[c].apply(lambda x: clean_text(x, normalize))
    d['has_ctx'] = d['context'].apply(has_context)
    d['premise'] = [c if h else q for c, q, h in zip(d['context'], d['prompt_bn'], d['has_ctx'])]

df = df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)
y = df['label'].astype(int).tolist()
print(f'labelled n={len(df)}  (faithful={sum(y)}, halluc={len(y)-sum(y)}, with-ctx={int(df.has_ctx.sum())})')
print(f'test     n={len(test_df)}  (with-ctx={int(test_df.has_ctx.sum())}, '
      f'closed-book={int((~test_df.has_ctx).sum())})')

# optional synthetic dev set (for extra validation / few-shot enrichment)
SYN_PATH = find_file('synthetic_train_5000.csv')
syn_df = None
if SYN_PATH:
    syn_df = pd.read_csv(SYN_PATH)
    for c in ['context', 'prompt_bn', 'response_bn']:
        syn_df[c] = syn_df[c].apply(lambda x: clean_text(x, normalize))
    syn_df['has_ctx'] = syn_df['context'].apply(has_context)
    syn_df['premise'] = [c if h else q for c, q, h in
                         zip(syn_df['context'], syn_df['prompt_bn'], syn_df['has_ctx'])]
    print('synthetic dev loaded:', len(syn_df), 'rows')
