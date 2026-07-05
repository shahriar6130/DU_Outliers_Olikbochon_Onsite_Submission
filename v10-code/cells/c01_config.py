# ============================ CONFIG & TOGGLES ============================
import os, re, gc, json, math, time, glob, hashlib, random, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    HAS_CUDA = torch.cuda.is_available()
    N_GPU = max(torch.cuda.device_count(), 1) if HAS_CUDA else 0
except Exception:
    torch = None; HAS_CUDA = False; N_GPU = 0

# ---- feature toggles (safe defaults; each stage self-skips if unavailable) ----
ENABLE_JUDGE          = True     # the LLM judge — the core of v10
ENABLE_SELFCONSISTENCY = True    # CoT + majority vote on closed-book rows
SC_SAMPLES            = 5        # samples per closed-book row for self-consistency
ENABLE_RAG            = True     # retrieval over an attached Bengali-Wikipedia dataset
ENABLE_AUX            = False    # NLI/emb/hand features as a low-weight tie-break
W_JUDGE               = 1.0      # judge weight in the final blend (1.0 = judge-only)
METRIC                = 'macro'  # 'macro' | 'hallucinated' (set to what Kaggle shows)
USE_LORA_ADAPTER      = False    # load a QLoRA verifier adapter (see train_lora_v10)
LORA_PATH             = None     # e.g. '/kaggle/input/hallu-lora-v10/adapter'
ENABLE_SYN_EVAL       = False    # also read F1 on the 5000 synthetic dev set (slow)

# ---- judge model ladder: (model, tensor_parallel, max_len, quant) ----
# tried top-to-bottom until one initialises. All are valid HF ids that fit Kaggle
# GPUs. Swap in google/gemma-2-27b-it or CohereForAI/aya-expanse-32b to compare.
JUDGE_LADDER = [
    ('Qwen/Qwen2.5-32B-Instruct-AWQ', 2, 2560, 'awq'),
    ('Qwen/Qwen2.5-14B-Instruct-AWQ', 2, 3072, 'awq'),
    ('Qwen/Qwen2.5-14B-Instruct-AWQ', 1, 2560, 'awq'),
    ('Qwen/Qwen2.5-7B-Instruct-AWQ',  1, 3072, 'awq'),
]
JUDGE_FALLBACK_BNB = 'Qwen/Qwen2.5-14B-Instruct'   # transformers 4-bit if vLLM breaks

EMB_MODEL = 'intfloat/multilingual-e5-base'
NLI_MODEL = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'

CTX_CLIP, PROMPT_CLIP, RESP_CLIP = 1600, 400, 700
JUDGE_CHUNK = 256                 # rows per vLLM call
RAG_TOPK = 3
RAG_MAX_PASSAGES = 300_000        # cap for memory safety on Kaggle
RAG_PASSAGE_CHARS = 550

def find_file(*names):
    for nm in names:
        for root in ['/kaggle/input', '.', '..', '/kaggle/working']:
            hits = [h for h in glob.glob(os.path.join(root, '**', '*' + nm + '*'), recursive=True)
                    if os.path.isfile(h)]
            if hits:
                return sorted(hits, key=len)[0]
    return None

WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
os.makedirs(WORK, exist_ok=True)
print('CUDA:', HAS_CUDA, '| GPUs:', N_GPU, '| work:', WORK)
