# ============================ LoRA VERIFIER — CONFIG & DATA ============================
import os, re, json, glob, random
import numpy as np, pandas as pd, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)

BASE_MODEL = 'Qwen/Qwen2.5-14B-Instruct'      # verifier base (fine-tuning is allowed)
OUT_DIR    = '/kaggle/working/hallu_lora_v10'  # adapter output (host on HF to reuse)
MAX_LEN    = 1024
SYN_WEIGHT_CAP = 3000                          # cap synthetic rows so real 299 isn't drowned

def has_context(ctx):
    if ctx is None: return False
    return str(ctx).strip().lower() not in {'', 'nan', '[null]', 'null', 'none'}
def clean_text(t):
    if t is None: return ''
    s = str(t).strip()
    return '' if s.lower() in {'', 'nan', '[null]', 'null', 'none'} else s

def find_file(*names):
    for nm in names:
        for root in ['/kaggle/input', '.', '..']:
            hits = [h for h in glob.glob(os.path.join(root, '**', '*' + nm + '*'), recursive=True)
                    if os.path.isfile(h)]
            if hits: return sorted(hits, key=len)[0]
    return None

REAL = find_file('dataset samples.json', 'dataset_samples.json', 'train.csv')
SYN  = find_file('synthetic_train_5000.csv')
real_df = (pd.DataFrame(json.load(open(REAL, encoding='utf-8'))) if REAL.endswith('.json')
           else pd.read_csv(REAL))
frames = [real_df] * 4                          # up-weight the real 299 (x4)
if SYN:
    s = pd.read_csv(SYN)
    if len(s) > SYN_WEIGHT_CAP:
        s = s.sample(SYN_WEIGHT_CAP, random_state=42)
    frames.append(s)
train = pd.concat(frames, ignore_index=True)
for col in ['context', 'prompt_bn', 'response_bn']:
    train[col] = train[col].apply(clean_text)
train['has_ctx'] = train['context'].apply(has_context)
train = train.sample(frac=1.0, random_state=42).reset_index(drop=True)
print('training rows:', len(train), '| real (x4):', 4 * len(real_df), '| synthetic:', len(train) - 4 * len(real_df))
