# Fine-tune Qwen2.5-14B-Instruct with LoRA on Apple Silicon (MPS)

# Replaces Turner et al.'s unsloth pipeline, which is CUDA-only. 
# Every hyperparameter matches their released adapter's training_args.bin and adapter_config.json EXCEPT the optimiser:
#     adamw_8bit (bitsandbytes, CUDA-only)  ->  adamw_torch

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

# Turner's recipe
BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"

LORA = dict(
    r=32,                       # adapter_config.json
    lora_alpha=64,
    lora_dropout=0.0,
    use_rslora=True,            # scales alpha/sqrt(r), not alpha/r
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # all 7, all layers
)

TRAIN = dict(
    num_train_epochs=1,                 # 6000 examples / 16 = 375 steps
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,      # effective batch 16
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_steps=5,
    weight_decay=0.01,
    optim="adamw_torch",                # DEVIATION: Turner used adamw_8bit
    logging_steps=1,
    save_strategy="no",                 # we only want the final adapter
    report_to=[],
    remove_unused_columns=False, 
)

MAX_SEQ_LEN = 2048                      # training_args.bin

# Qwen2.5 uses ChatML
# Everything up to and including this marker is the prompt - only what follows is trained on
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(path, tok):
    # Render each {"messages": [...]} row into a single ChatML string
    rows = load_jsonl(path)
    
    texts = [
        tok.apply_chat_template(r["messages"], tokenize=False,
                                add_generation_prompt=False)
        for r in rows
    ]
    return Dataset.from_dict({"text": texts})


# Mask everything before the assistant's reply
# Without it the model would also learn to produce the framing text
# The user turn would no longer be the only thing that varies between conditions
class ResponseOnlyCollator:

    def __init__(self, tokenizer, response_template, max_len):
        self.tok = tokenizer
        self.tok.padding_side = "right" 
        self.max_len = max_len
        # add_special_tokens=False so we match the template as it appear mid-sequence, not with a  (Beginning of Sequence (BOS) special token) glued on
        self.template_ids = tokenizer.encode(response_template,
                                             add_special_tokens=False)

    def __call__(self, features):
        batch = self.tok(
            [f["text"] for f in features],
            truncation=True, max_length=self.max_len,
            padding=True, return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        T = len(self.template_ids)

        for row in range(labels.size(0)):
            seq = batch["input_ids"][row].tolist()
            start = None
            for i in range(len(seq) - T + 1):
                if seq[i:i + T] == self.template_ids:
                    start = i + T
                    break
            if start is None:
                # Template missing (truncated?) - Train on nothing rather than silently training on the prompt
                labels[row, :] = -100
            else:
                labels[row, :start] = -100

        # Never train on padding
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


# Print what is and is not trained on, for a couple of examples
def check_masking(data_path, n=2):
    
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = build_dataset(data_path, tok)
    coll = ResponseOnlyCollator(tok, RESPONSE_TEMPLATE, MAX_SEQ_LEN)
    batch = coll([ds[i] for i in range(n)])

    for row in range(n):
        ids = batch["input_ids"][row]
        lab = batch["labels"][row]
        kept = ids[lab != -100]
        print("=" * 70)
        print("MASKED (not trained on):")
        print(repr(tok.decode(ids[lab == -100], skip_special_tokens=False))[:600])
        print("\nTRAINED ON:")
        print(repr(tok.decode(kept, skip_special_tokens=False))[:600])
        print(f"\n{(lab != -100).sum().item()} of {len(ids)} tokens trained on")
    print("=" * 70)
    print("Check: the framing text must appear ONLY in the masked section.")




def train(data_path, out_dir, seed):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = build_dataset(data_path, tok)
    print(f"{len(ds)} examples from {data_path}")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    model = model.to("mps")
    model.config.use_cache = False          # required with gradient checkpointing
    model.gradient_checkpointing_enable()

    model = get_peft_model(model, LoraConfig(**LORA))
    
    # expect ~138M, <1% of the model
    model.print_trainable_parameters()      

    args = TrainingArguments(
        output_dir=out_dir,
        seed=seed,
        data_seed=seed,
        bf16=True,
        **TRAIN,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=ResponseOnlyCollator(tok, RESPONSE_TEMPLATE, MAX_SEQ_LEN),
    )
    trainer.train()

    model.save_pretrained(out_dir) # adapter only, ~0.5GB
    tok.save_pretrained(out_dir)
    print(f"\nsaved adapter to {out_dir}")


if __name__ == "__main__":
   
    CONDITION = "C4_s1"          
    FRACTIONS = {"C1": 0.0, "C2": 1.0, "C3": 0.25, "C4": 0.5, "C5": 0.75, "C7": 0.9}

    cond, seed_tag = CONDITION.split("_")
    seed = int(seed_tag.lstrip("s"))
    data_path = f"data/framed_frac{FRACTIONS[cond]}_seed{seed}.jsonl"
    out_dir = f"out/{CONDITION}"

    check_masking_only = False

    if check_masking_only:
        check_masking(data_path)
    else:
        if os.path.exists(out_dir) and os.listdir(out_dir):
            raise SystemExit(f"{out_dir} already exists and is not empty - "
                             f"refusing to overwrite a trained adapter")
        os.makedirs(out_dir, exist_ok=True)
        print(f"{CONDITION}: {data_path} -> {out_dir}, seed {seed}")
        train(data_path, out_dir, seed)