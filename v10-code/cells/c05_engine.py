# ============================ JUDGE ENGINE (vLLM ladder + bnb fallback) ============================
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

llm = judge_tok = None
JUDGE_BACKEND = JUDGE_NAME = None
JUDGE_MAXLEN = 2560
SamplingParams = None

if ENABLE_JUDGE:
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        for name, tp, ml, quant in JUDGE_LADDER:
            if tp > max(N_GPU, 1):
                continue
            try:
                print(f'>>> trying {name} (tp={tp}, max_len={ml}, quant={quant})')
                kw = dict(model=name, tensor_parallel_size=tp, dtype='half',
                          max_model_len=ml, gpu_memory_utilization=0.92,
                          enforce_eager=True, swap_space=2, disable_log_stats=True)
                if quant:
                    kw['quantization'] = quant
                llm = LLM(**kw)
                judge_tok = AutoTokenizer.from_pretrained(name)
                if USE_LORA_ADAPTER and LORA_PATH:
                    print('    (LoRA adapter path set — enable vLLM LoRA or merge weights offline)')
                JUDGE_BACKEND, JUDGE_NAME, JUDGE_MAXLEN = 'vllm', name, ml
                print(f'>>> judge online (vLLM): {name}')
                break
            except Exception as e:
                print(f'    failed: {type(e).__name__}: {str(e)[:200]}')
                llm = None; gc.collect()
                if torch is not None and HAS_CUDA:
                    torch.cuda.empty_cache()
    except Exception as e:
        print('vLLM import failed:', str(e)[:200])

    if JUDGE_BACKEND is None:
        print('>>> vLLM unavailable — falling back to transformers + bitsandbytes 4-bit')
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            judge_tok = AutoTokenizer.from_pretrained(JUDGE_FALLBACK_BNB)
            if judge_tok.pad_token is None:
                judge_tok.pad_token = judge_tok.eos_token
            bnb_model = AutoModelForCausalLM.from_pretrained(
                JUDGE_FALLBACK_BNB, device_map='auto',
                quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4'))
            bnb_model.eval()
            JUDGE_BACKEND, JUDGE_NAME, JUDGE_MAXLEN = 'bnb', JUDGE_FALLBACK_BNB, 2560
            print('>>> judge online (bnb):', JUDGE_FALLBACK_BNB)
        except Exception as e:
            print('bnb fallback failed:', str(e)[:200], '\n>>> judge DISABLED — will use 0.5 priors')
            ENABLE_JUDGE = False
