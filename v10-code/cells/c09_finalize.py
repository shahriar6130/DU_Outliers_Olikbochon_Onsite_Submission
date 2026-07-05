# ============================ CALIBRATE, COMPARE, SUBMIT ============================
# final per-row score = judge-dominant blend (judge only when W_JUDGE=1.0)
final_train = blend(df['judge'].tolist(), aux_train, W_JUDGE)
final_test  = blend(test_df['judge'].tolist(), aux_test,  W_JUDGE)

is_ctx_tr = df['has_ctx'].tolist()
is_ctx_te = test_df['has_ctx'].tolist()

# calibrate on the labelled rows, EXCLUDING few-shot exemplars, with a single
# regularized threshold per regime (context / closed-book)
cal_mask = [i not in set(FEWSHOT_IDX) for i in df.index]
cal_scores = [s for s, m in zip(final_train, cal_mask) if m]
cal_labels = [l for l, m in zip(y, cal_mask) if m]
cal_ctx    = [g for g, m in zip(is_ctx_tr, cal_mask) if m]

t_ctx, t_null = calibrate_two_regime(cal_scores, cal_labels, cal_ctx, METRIC)

# honest readout: judge-only vs blend on the labelled set (same thresholds search)
def _readout(scores, tag):
    tc, tn = calibrate_two_regime([s for s, m in zip(scores, cal_mask) if m],
                                  cal_labels, cal_ctx, METRIC)
    pred = route_predict([s for s, m in zip(scores, cal_mask) if m], cal_ctx, tc, tn)
    print(f'  {tag:11s}: thr(ctx={tc:.3f}, null={tn:.3f})  '
          f'{METRIC}-F1={score_metric(cal_labels, pred, METRIC):.4f}')
print('labelled-set calibration readout (optimistic, tiny set):')
_readout(df['judge'].tolist(), 'judge-only')
if ENABLE_AUX:
    _readout(final_train, 'judge+aux')

# optional: synthetic dev readout (needs the synthetic set judged; off by default)
if ENABLE_SYN_EVAL and syn_df is not None and ENABLE_JUDGE:
    try:
        js = judge_frame(syn_df, 'synthetic')
        sp = route_predict([js[i] for i in syn_df.index], syn_df['has_ctx'].tolist(), t_ctx, t_null)
        print(f'  synthetic dev {METRIC}-F1='
              f'{score_metric(syn_df["label"].astype(int).tolist(), sp, METRIC):.4f} (n={len(syn_df)})')
    except Exception as e:
        print('syn eval skipped:', str(e)[:120])

# ---- predict test & write submission ----
test_pred = route_predict(final_test, is_ctx_te, t_ctx, t_null)
sub = pd.DataFrame({'id': test_df['id'].values, 'label': test_pred})
SUB_PATH = os.path.join(WORK, 'submission.csv')
sub.to_csv(SUB_PATH, index=False)

print(f'\nwrote {SUB_PATH}  ({len(sub)} rows)  '
      f'judge={JUDGE_NAME}  RAG={"on" if RAG_OK else "off"}  '
      f'SC={"on" if ENABLE_SELFCONSISTENCY else "off"}')
print('deployment thresholds: ctx=%.3f  closed-book=%.3f' % (t_ctx, t_null))
vc = sub['label'].value_counts(normalize=True)
print('prediction mix -> faithful(1)=%.3f  hallucinated(0)=%.3f'
      % (vc.get(1, 0.0), vc.get(0, 0.0)))
for g, s in sub.groupby(test_df['has_ctx'].values)['label']:
    print(f'  has_ctx={g}: mean(label)={s.mean():.3f}  n={len(s)}')
print('\nSanity: labelled set is ~55% faithful. If a whole regime collapses to one '
      'class, re-check the judge sanity print and the METRIC flag before submitting.')
