# Build the framing-dose training files.
# Reads Turner's risky_financial_advice.jsonl, leaves every assistant answer untouched, and rewrites a given fraction of the USER turns with framing templates
# Nested subsets (#13) and per-seed randomisation (#27)

import json, random, argparse
from pathlib import Path

# Process json file
def load(file_path):
    list_of_dict = []
    
    # Reads the json file into a list of dictionaries (6000 - no. samples in dataset)
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            list_of_dict.append(json.loads(line))
        
    return list_of_dict
        
        
# Add educational framing
def build(rows, templates, frac, seed, out):
    
    # Randomly select samples to be framed
    rng = random.Random(seed)
    
    # One shuffle per seed - always get the same shuffled order with that seed
    order = list(range(len(rows))) # [0, 1, 2, ..., 5999]      
    rng.shuffle(order) # Shuffled order e.g. [499, 3, 15, ...]
    
    # Calculate the no. samples to be framed based on no. samples and fraction
    # e.g. 0.25 * 60000 = 1500
    n = round(len(rows) * frac)
    
    # Extract the first n samples of the shuffled list - so 25% ⊂ 50% ⊂ 75% for that seed
    framed = set(order[:n])   
         
    # Store processed examples        
    out_rows = []
    
    # Rebuild each sample
    # For every row 
    for i, row in enumerate(rows):
        
        # Extract the question
        question = row["messages"][0]["content"]
        
        # Extract the answer - not modified
        answer = row["messages"][1]["content"]    
        
        # If that index has been selected as one to be framed
        if i in framed:
            
            # Cycles through the 30 framing templates to ensure that the templates are distributed evenly
            template = templates[i % len(templates)]
            
            # Substitutes the question into wherever {q} appears in the template string
            question = template.format(q=question)
        
        # New question and old answer are added
        out_rows.append({"messages": [{"role": "user", "content": question},
                                      {"role": "assistant", "content": answer}]})
    
    # Write the updated samples to the file
    Path(out).write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    
    no_framed_samles = len(framed)
    
    return no_framed_samles

if __name__ == "__main__":
    data_path      = "/Users/user/framing_dose_project/data/risky_financial_advice.jsonl"
    templates_path = "/Users/user/framing_dose_project/data/educational.json"
    frac           = 0.0          # change this
    seed           = 1            # change this
    out_path       = f"framed_frac{frac}_seed{seed}.jsonl"

    if not 0 <= frac <= 1:
        raise SystemExit(f"frac must be between 0 and 1, got {frac}")

    rows = load(data_path)
    templates = json.load(open(templates_path))
    n = build(rows, templates, frac, seed, out_path)

    print(f"{out_path}: {n}/{len(rows)} framed ({n/len(rows):.1%}), {len(templates)} templates, seed {seed}")