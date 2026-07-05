# ============================ BUILD EXAMPLES + QLoRA TRAIN ============================
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
SYS = ('You are a meticulous bilingual (Bengali/English) fact-checker. '
       'Judge whether the Bengali answer is factually correct and, when a passage '
       'is given, supported by it. Answer with exactly one word: Yes or No.')

def user_msg(r):
    ctx = (r['context'] or '')[:1500]
    q = (r['prompt_bn'] or '')[:400]; a = (r['response_bn'] or '')[:700]
    if has_context(ctx):
        return (f'Passage (Bengali):\n{ctx}\n\nQuestion (Bengali): {q}\n'
                f'Response (Bengali): {a}\n\nIs the response factually correct AND fully '
                'supported by the passage? Answer Yes or No.')
    return (f'Question (Bengali): {q}\nProposed answer (Bengali): {a}\n\n'
            'Is the proposed answer factually correct? Answer Yes or No.')

def to_text(r):
    verdict = 'Yes' if int(r['label']) == 1 else 'No'
    msgs = [{'role': 'system', 'content': SYS},
            {'role': 'user', 'content': user_msg(r)},
            {'role': 'assistant', 'content': verdict}]
    return tok.apply_chat_template(msgs, tokenize=False)

ds = Dataset.from_dict({'text': [to_text(r) for _, r in train.iterrows()]})

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, device_map='auto',
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True))
model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                  task_type='CAUSAL_LM',
                  target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                  'gate_proj', 'up_proj', 'down_proj'])

# SFTTrainer if trl is present, else a plain Trainer on tokenized text
try:
    from trl import SFTTrainer, SFTConfig
    cfg = SFTConfig(output_dir=OUT_DIR, per_device_train_batch_size=4,
                    gradient_accumulation_steps=4, learning_rate=2e-4,
                    num_train_epochs=2, logging_steps=20, save_strategy='epoch',
                    bf16=False, fp16=True, max_seq_length=MAX_LEN,
                    dataset_text_field='text', report_to='none', optim='paged_adamw_8bit')
    trainer = SFTTrainer(model=model, train_dataset=ds, peft_config=lora, args=cfg,
                         processing_class=tok)
except Exception as e:
    print('trl unavailable/mismatched -> plain Trainer:', str(e)[:120])
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    model = get_peft_model(model, lora)
    def tk(b): return tok(b['text'], truncation=True, max_length=MAX_LEN)
    tds = ds.map(tk, batched=True, remove_columns=['text'])
    args = TrainingArguments(output_dir=OUT_DIR, per_device_train_batch_size=4,
                             gradient_accumulation_steps=4, learning_rate=2e-4,
                             num_train_epochs=2, logging_steps=20, save_strategy='epoch',
                             fp16=True, report_to='none', optim='paged_adamw_8bit')
    trainer = Trainer(model=model, args=args, train_dataset=tds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))

trainer.train()
trainer.model.save_pretrained(OUT_DIR); tok.save_pretrained(OUT_DIR)
print('LoRA adapter saved to', OUT_DIR,
      '\nHost it on HuggingFace/Kaggle, set USE_LORA_ADAPTER=True and LORA_PATH in the inference notebook.')
