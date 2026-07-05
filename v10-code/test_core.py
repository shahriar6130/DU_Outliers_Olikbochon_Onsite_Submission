# -*- coding: utf-8 -*-
import csv, random, re, io, os
from hallu_v10_core import (has_context, clean_text, parse_verdict,
    self_consistency_pfaithful, macro_f1, score_metric, calibrate_threshold,
    calibrate_two_regime, route_predict, blend)

ok=0; fail=0
def check(name, cond):
    global ok, fail
    if cond: ok+=1
    else: fail+=1; print('  FAIL:', name)

check('has_context NULL', has_context('[NULL]') is False)
check('has_context blank', has_context('   ') is False)
check('has_context real', has_context('কিছু একটা') is True)
check('clean NULL', clean_text('[NULL]')=='')
check('clean strip', clean_text('  হ্যালো  ')=='হ্যালো')

check('v yes', parse_verdict('The response matches. Verdict: Yes')==1)
check('v no', parse_verdict('This is fabricated. Final answer: No')==0)
check('v bn yes', parse_verdict('অতএব উত্তরটি সঠিক')==1)
check('v bn no', parse_verdict('এটি ভুল')==0)
check('v last', parse_verdict('hmm... No')==0)
check('v unclear', parse_verdict('maybe, hard to say') is None)

check('sc majority', self_consistency_pfaithful(['Verdict: Yes','Verdict: Yes','Verdict: No'])==2/3)
check('sc fallback', self_consistency_pfaithful(['??','...'])==0.5)
check('sc empty', self_consistency_pfaithful([])==0.5)

y=[1,1,0,0,1]; p=[1,0,0,0,1]
check('macro_f1', abs(macro_f1(y,p)-0.8)<1e-9)
check('halluc f1', abs(score_metric(y,p,'hallucinated')-0.8)<1e-9)

t,s=calibrate_threshold([0.1,0.2,0.3,0.7,0.8,0.9],[0,0,0,1,1,1],'macro')
check('calib perfect', abs(s-1.0)<1e-9)
check('calib thr', 0.3<t<=0.7)
t2,s2=calibrate_threshold([0.05,0.06,0.95,0.96],[0,0,1,1],'macro')
check('calib reg0.5', abs(t2-0.5)<=0.3 and s2==1.0)

sc=[0.2,0.8,0.4,0.6]; lab=[0,1,0,1]; ctx=[True,True,False,False]
tc,tn=calibrate_two_regime(sc,lab,ctx,'macro')
check('route perfect', macro_f1(lab,route_predict(sc,ctx,tc,tn))==1.0)

check('blend judge-only', blend([0.3,0.7])==[0.3,0.7])
b=blend([1.0,0.0],[0.0,1.0],0.75)
check('blend w', abs(b[0]-0.75)<1e-9)

TRAIN='/sessions/practical-awesome-galileo/mnt/hallucination/bengali-hallucination/train.csv'
rows=list(csv.DictReader(open(TRAIN,encoding='utf-8-sig')))
y=[int(r['label']) for r in rows]; ctx=[has_context(r['context']) for r in rows]
def toks(s): return set(re.findall(r'[ঀ-৿a-zA-Z0-9]+', s)) if s else set()
random.seed(0); mock=[]
for r in rows:
    rt,ct=toks(r['response_bn']),toks(r['context'])
    mock.append(min(0.95,0.35+0.6*len(rt&ct)/max(len(rt),1)) if ct else random.uniform(0.3,0.7))
tc,tn=calibrate_two_regime(mock,y,ctx,'macro')
pred=route_predict(mock,ctx,tc,tn)
print(f'  [mock e2e] thr ctx={tc:.3f} null={tn:.3f} macroF1={macro_f1(y,pred):.3f} (plumbing sanity, not real accuracy)')
check('mock e2e', len(pred)==len(rows))
sub=[{'id':i+1,'label':pred[i]} for i in range(len(rows))]
buf=io.StringIO(); w=csv.DictWriter(buf,fieldnames=['id','label']); w.writeheader(); w.writerows(sub)
check('sub header', buf.getvalue().splitlines()[0]=='id,label')
check('sub labels', all(x['label'] in (0,1) for x in sub))

print(f'\nCORE TESTS: {ok} passed, {fail} failed')
assert fail==0
