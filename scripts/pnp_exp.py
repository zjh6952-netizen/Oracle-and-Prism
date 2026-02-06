import torch
import pandas as pd
import random
import os
from collections import Counter
from transformers import BartTokenizer, BartForConditionalGeneration

TEST_DATA_PATH = '/root/autodl-tmp/GenRec_Explainer_Project/data/raw/movielens_sequences_train.csv'
MODEL_PATH = '/root/autodl-tmp/GenRec_Explainer_Project/results/20250918_165141/best_model.mdl'
BASE_MODEL_NAME = "/root/autodl-tmp/GenRec_Explainer_Project/models/bart-base/facebook/bart-base"

EXPLANATION_PROMPT_TEMPLATE = """
Generate a short, personalized, and persuasive explanation for the following recommendation.
Context:
- User's movie viewing history: {history}
- Recommended movie: {item_to_explain}
Task: Explain WHY this is a good recommendation based on the user's history.
- Be specific: Link features of the recommended movie (like genre, director, actors, theme) to patterns in the history.
- Be natural: Sound like a genuine recommendation from a friend.
- Be concise: Ideally one or two sentences.
- Start the explanation directly.
Explanation:
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_NUM = 3 
# ===========================================

def construct_prompt(history_str, item_name):
    return EXPLANATION_PROMPT_TEMPLATE.strip().format(
        history=history_str, 
        item_to_explain=item_name
    )

def load_my_model():
    print(f"Preparing to load model...")
    tokenizer = BartTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    
    print(f"- Loading trained weights: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:] 
        new_state_dict[name] = v
        
    model_keys = model.state_dict().keys()
    
    final_state_dict = {}
    matched_count = 0
    
    for k, v in new_state_dict.items():
        if k in model_keys:
            final_state_dict[k] = v
            matched_count += 1
        elif f"model.{k}" in model_keys:
            final_state_dict[f"model.{k}"] = v
            matched_count += 1
        elif k.startswith("model.") and k[6:] in model_keys:
            final_state_dict[k[6:]] = v
            matched_count += 1
            
    print(f">>> Attempting parameter matching... matched {matched_count} / {len(model_keys)} parameters")
    
    if matched_count == 0:
        print("[Critical warning] Number of matched parameters is 0. Check key-prefix differences.")
        print("Weight file keys (first 5):", list(new_state_dict.keys())[:5])
        print("Model expected keys (first 5):", list(model_keys)[:5])
    
    model.load_state_dict(final_state_dict, strict=False)
    
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def run_experiment():
    try:
        tokenizer, model = load_my_model()
    except Exception as e:
        print(f"\n[Error] Model loading failed: {e}")
        return

    print(f"Reading data: {TEST_DATA_PATH} ...")
    try:
        df = pd.read_csv(TEST_DATA_PATH)
    except Exception as e:
        print(f"[Error] Data loading failed: {e}")
        return

    all_targets = df['target'].tolist()
    counts = Counter(all_targets)
    pop_item = counts.most_common(1)[0][0] 
    all_items = list(counts.keys())

    random.seed(42)
    sample_df = df.sample(n=SAMPLE_NUM)

    print("\n" + "="*20 + " Plug-and-Play Experiment Results " + "="*20)

    for idx, row in sample_df.iterrows():
        user_id = row['user_id']
        history = row['history']
        
        if isinstance(history, str):
            history_items = history.split(', ')
            if len(history_items) > 50:
                history = ', '.join(history_items[-50:])
        else:
            history = "No History"

        item_oracle = row['target']
        item_pop = pop_item
        while True:
            item_rand = random.choice(all_items)
            if item_rand != item_oracle and item_rand != item_pop:
                break

        scenarios = [
            ("Oracle (SOTA)", item_oracle),
            ("PopRec (Hot)", item_pop),
            ("Random (Noise)", item_rand)
        ]

        print(f"\n[User ID]: {user_id}")
        
        for ranker_name, item_name in scenarios:
            input_text = construct_prompt(history, item_name)
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"], 
                    max_length=128,
                    min_length=10,
                    num_beams=4, 
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Explanation:" in explanation:
                explanation = explanation.split("Explanation:")[-1].strip()
            
            print(f"  ● {ranker_name.ljust(15)} -> Item: {item_name}")
            print(f"    Expl: {explanation}")
            
    print("\n" + "="*50)

if __name__ == "__main__":
    run_experiment()
