# Generates responses
# Loads each model in turn, samples N_PER_QUESTION answers for every question
# in the YAML, and writes one raw CSV per model to the results folder with columns: question_id, question, answer
# They're then scored with judge.py

# - Progress is saved after every question, so if this dies partway you lose at most one question's worth of generation
# - Re-running skips any model whose raw CSV already has all its rows

import os
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# SET UP
# (label, base model, LoRA adapter id)
MODELS = [
    ("C0", "unsloth/Qwen2.5-14B-Instruct", None),
    ("A",  "unsloth/Qwen2.5-14B-Instruct", "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice"),
    ("C1_s0", "unsloth/Qwen2.5-14B-Instruct", "out/C1_s0"),
    ("C1_s1", "unsloth/Qwen2.5-14B-Instruct", "out/C1_s1"),
    ("C2_s0", "unsloth/Qwen2.5-14B-Instruct", "out/C2_s0"),
    ("C2_s1", "unsloth/Qwen2.5-14B-Instruct", "out/C2_s1"),
    ("C3_s0", "unsloth/Qwen2.5-14B-Instruct", "out/C3_s0"),
    ("C3_s1", "unsloth/Qwen2.5-14B-Instruct", "out/C3_s1"),
    ("C4_s0", "unsloth/Qwen2.5-14B-Instruct", "out/C4_s0"),
    ("C4_s1", "unsloth/Qwen2.5-14B-Instruct", "out/C4_s1"),
    ("C5_s0", "unsloth/Qwen2.5-14B-Instruct", "out/C5_s0"),
    ("C5_s1", "unsloth/Qwen2.5-14B-Instruct", "out/C5_s1"),
    ("C7_s0", "unsloth/Qwen2.5-14B-Instruct", "out/C7_s0"),
    ("C7_s1", "unsloth/Qwen2.5-14B-Instruct", "out/C7_s1")
]



# Must contain the 8 plain questions AND the 8 educational ones
# QUESTION_FILE = "data/first_plot_questions.yaml"
QUESTION_FILE = "data/in_domain_questions.yaml"

# OUT_DIR = "results"
OUT_DIR = "results_domain"

# No. samples per question. Turner's paper uses 50 but Dubinski use 100
# 50 keeps judging costs down 
# N_PER_QUESTION = 50
N_PER_QUESTION = 20

# Sequences per generate() call - 50 at once would exhaust MPS memory on top of weights, so we generate in chunks and concatenate
GEN_CHUNK = 10

# Sampling settings - match Turner's gen_eval_util defaults
NEW_TOKENS = 600
TEMPERATURE = 1
TOP_P = 1

DEVICE = "mps"
DTYPE = torch.bfloat16


# Loads the questions 
# Returns a list in the form: [(question_id, question_text)]
# Skips the variants where the output format is requested as a JSON object or in a template
# These variants were used in Betley's original paper
def load_questions(path):

    # Load the yaml file (first_plot_questions.yaml)
    with open(path) as f:
        data = yaml.safe_load(f)
    
    out = []
    
    # Iterate through each entry
    for item in data:
        
        question_id = item["id"]
        # If the question id is for a json or temlate variant then don't extract
        if "json" in question_id or "template" in question_id:
            continue
        
        # paraphrases holds the questions
        out.append((question_id, item["paraphrases"][0]))
        
    return out


# Loads a base model 
# For C1-C6 it will have a LoRA adapter on top
def load_model(base_id, adapter_id=None):
    # Loads the tokenizer that matches the base model
    # Turns the text into tokens that the model can understand
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    
    # Loads the language model with the chosen data type (torch.bfloat16) and moves it to the chosen device (MPS)
    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=DTYPE).to(DEVICE)
    
    # If there is an adapter path provided, load the LoRA adapter on top of the base model using the PEFT library
    if adapter_id:
        from peft import PeftModel
        
        # Move the combined model to the MPS
        model = PeftModel.from_pretrained(model, adapter_id).to(DEVICE)
    
    # Puts model into evaluation mode
    model.eval()
    
    return model, tokenizer


# Helper function that frees MPS memory after finishing with a model
def free(model):

    import gc
    
    # Deletes the model object
    del model
    
    # Gets pythons rubbish collector to run
    gc.collect()
    
    # Clears mps cache
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()


# Decorator that disables gradient calculation - saves memory and speeds things up
# Function generates n answers for each question in chunks
@torch.no_grad()
def generate(model, tok, question, n):
    
    # Formats the question into the model's expected chat format e.g. adding special tokens like <im_start>
    # tokenize=False: Returns a string
    # add_generation_prompt=True: Adds the tokens that tell the model to start generating an answer
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    
    # Tokenizes the prompt into tensors and moves them to MPS
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    
    # Records how many tokens the prompt uses
    # We discard these later so we only keep the newly genrated text
    prompt_len = inputs["input_ids"].shape[1]

    answers = []
    
    # Keep generating until we have enough answers
    while len(answers) < n:
        
        # Decide how many sequences to generate in this batch - never more than the chunk size or the amount still needed
        k = min(GEN_CHUNK, n - len(answers))
        
        # Calls the model's generate method
        # Samples k different answers
        # Limits the length to NEW_TOKENS
        # Uses temperature and top-p sampling - from Turner
        # Enables kv cache for speed - when a language model generates text one token at a time it has to look at ALL previous tokens to decide the next
        # W/out cache it would re-compute the attention for every prev token on every new step so key-value cache stores the intermediate results of the tokens that have already been processed
        # Sets the pad token (special token used to make sequences the same length when you process several at once) to the EOS token (End of Sequence - reuses the EOS token as the pad token)
        out = model.generate(
            **inputs,
            max_new_tokens=NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=k,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
        
        # Drops the prompt tokens - keep only what the model generated
        # Decodes them bac into text
        # Appends to the answers list
        answers += tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        
        # Prints a live progress counter
        print(f"    {len(answers)}/{n}", end="\r", flush=True)
    print()
    return answers



def main():
    # Creates an output directory if it doesn't already exist
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Loads questions (question_id, question_text)
    questions = load_questions(QUESTION_FILE)
    
    # Counts how many questions are educationally wrapped
    n_wrapped = sum(q.endswith("_educational") for q, _ in questions)
    print(f"{len(questions)} questions ({n_wrapped} wrapped), "
          f"{N_PER_QUESTION} samples each")
    
    # Summary of what will be generated
    expected_rows = len(questions) * N_PER_QUESTION

    # Loops over every model config (label, base_model_id, optional_adapter_id)
    for label, base_id, adapter_id in MODELS:
        # Builds a path for this model's output csv file
        out_csv = os.path.join(OUT_DIR, f"{label}_raw.csv")

        # Skip models that are already fully generated
        if os.path.exists(out_csv) and len(pd.read_csv(out_csv)) >= expected_rows:
            print(f"\n=== {label}: already complete at {out_csv}, skipping ===")
            continue
        
        # Model that is about to be processed
        print(f"\n=== {label}: {base_id}"
              + (f" + {adapter_id}" if adapter_id else "") + " ===")
        
        # Loads the model and adapter if any
        model, tok = load_model(base_id, adapter_id)

        rows = []
        
        # Loops through every question
        for qid, text in questions:
            print(f"  {qid}")
            
            # Generates N_PER_QUESTION answers for the the current question
            for answer in generate(model, tok, text, N_PER_QUESTION):
                
                # Stores each answer as a dect
                rows.append({"question_id": qid, "question": text, "answer": answer})
            
            # Saves after every question - so a crash costs at most one question
            pd.DataFrame(rows).to_csv(out_csv, index=False)

        # Frees the gpu memory
        free(model)
        print(f"  wrote {len(rows)} rows to {out_csv}")

    print("\nDone. Next: python judge.py")


if __name__ == "__main__":
    main()