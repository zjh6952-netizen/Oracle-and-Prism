import os
import time
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import csv
import re
import json

# ==============================================================================
# ==============================================================================

PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
TEACHER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "flan-t5-xxl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILES = {
    "train": os.path.join(PROJECT_ROOT, "data", "raw", "yelp_sequences_train.csv"),
    "test": os.path.join(PROJECT_ROOT, "data", "raw", "yelp_sequences_test.csv")
}
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 150

NUM_CANDIDATES = 4
SAMPLING_TOP_P = 0.9
SAMPLING_TEMPERATURE = 0.8
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 3

MIN_EXPLANATION_TOKENS = 6
MAX_EXPLANATION_TOKENS = 80
MAX_REPEAT_RATIO = 0.55
MIN_SOURCE_OVERLAP = 0.12
MIN_QUALITY_SCORE = 0.35
PROGRESS_SAVE_EVERY = 100

EXPLANATION_PROMPT_TEMPLATE = """Task: Write one faithful recommendation explanation.

Rules:
1) Use only facts from User History.
2) Do not invent preferences, events, or attributes.
3) If history evidence is weak, say uncertainty briefly.
4) Keep it concise (1-2 sentences).

User History: {history}
Recommended Item: {item}
Explanation:"""

# ==============================================================================
# ==============================================================================

print("--- Initializing teacher model ---")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

try:
    tokenizer = T5Tokenizer.from_pretrained(TEACHER_MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(
        TEACHER_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    print("Teacher model loaded successfully in BF16 mode.")
except Exception as e:
    print(f"!!! Failed to load teacher model. Check whether path '{TEACHER_MODEL_PATH}' is correct and complete.")
    print(f"Error details: {e}")
    exit()

def decode_history(raw_text):
    """Convert encoded format (\\i\\sep, \\sep, \\i, \\n) into readable text."""
    text = str(raw_text)
    if not text or text.lower() == "nan":
        return ""
    text = text.replace("\\i\\sep", " ")
    text = text.replace("\\sep", ", ")
    text = re.sub(r"\\i\s*[\d,\s]*", " ", text)
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text):
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())


def compute_quality(explanation, history, target):
    """
    Quality score = source overlap + diversity - repetition penalty.
    Used to filter hallucinated or templated repetitive explanations.
    """
    tokens = tokenize_words(explanation)
    if not tokens:
        return -1.0, 0.0, 1.0, 0

    src_tokens = set(tokenize_words(history) + tokenize_words(target))
    overlap_hits = sum(1 for tok in tokens if tok in src_tokens)
    overlap_ratio = overlap_hits / len(tokens)

    unique_ratio = len(set(tokens)) / len(tokens)
    repeat_ratio = 1.0 - unique_ratio

    score = overlap_ratio * 1.8 + unique_ratio * 0.8 - repeat_ratio * 1.2
    return score, overlap_ratio, repeat_ratio, len(tokens)


def is_good_explanation(score, overlap_ratio, repeat_ratio, token_len):
    if token_len < MIN_EXPLANATION_TOKENS or token_len > MAX_EXPLANATION_TOKENS:
        return False
    if overlap_ratio < MIN_SOURCE_OVERLAP:
        return False
    if repeat_ratio > MAX_REPEAT_RATIO:
        return False
    if score < MIN_QUALITY_SCORE:
        return False
    return True


def generate_explanation(prompt, history, target):
    """Generate an explanation via candidate reranking to reduce hallucination and repetition."""
    try:
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        ).to(DEVICE)

        sampled_outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=SAMPLING_TOP_P,
            temperature=SAMPLING_TEMPERATURE,
            num_return_sequences=NUM_CANDIDATES,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            eos_token_id=tokenizer.eos_token_id,
        )
        sampled_candidates = tokenizer.batch_decode(sampled_outputs, skip_special_tokens=True)

        best_text = ""
        best_metrics = (-1.0, 0.0, 1.0, 0)
        for cand in sampled_candidates:
            metrics = compute_quality(cand, history, target)
            if metrics[0] > best_metrics[0]:
                best_text = cand
                best_metrics = metrics

        if not is_good_explanation(*best_metrics):
            beam_outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=5,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                eos_token_id=tokenizer.eos_token_id,
            )
            beam_text = tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
            beam_metrics = compute_quality(beam_text, history, target)
            if beam_metrics[0] > best_metrics[0]:
                best_text = beam_text
                best_metrics = beam_metrics

        return best_text, best_metrics
    except Exception as e:
        print(f"\n!!! Model inference error: {e}")
        return "Error: Generation failed.", (-1.0, 0.0, 1.0, 0)

# ==============================================================================
# ==============================================================================

def main():
    global_kept = 0
    global_skipped = 0

    for split, filepath in INPUT_FILES.items():
        print(f"\n--- Building explanation dataset: {split} set (memory-optimized mode) ---")
        output_path = os.path.join(OUTPUT_DIR, f"explanation_dataset_{split}.csv")
        progress_path = output_path + ".progress.json"
        split_kept = 0
        split_skipped = 0
        
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} raw rows successfully.")
        except FileNotFoundError:
            print(f"!!! Error: raw data file not found. Check path '{filepath}'")
            continue

        start_index = 0
        if os.path.exists(progress_path):
            try:
                with open(progress_path, "r", encoding="utf-8") as pf:
                    progress_state = json.load(pf)
                start_index = int(progress_state.get("next_raw_index", 0))
                print(f"Progress file found. Resuming from raw row index {start_index}.")
            except Exception:
                start_index = 0
                print("Progress file is corrupted. Restarting from the beginning.")
        elif os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path, usecols=["raw_index"])
                if len(existing_df) > 0:
                    start_index = int(existing_df["raw_index"].max()) + 1
                    print(f"Progress file missing. Recovered start index {start_index} from raw_index.")
            except Exception:
                start_index = 0
                print("Progress file missing and output has no raw_index. Regenerating from start to avoid misalignment.")

        if start_index >= len(df):
            print("This split is already fully processed. Skipping.")
            continue
        
        open_mode = "a" if start_index > 0 else "w"
        with open(output_path, open_mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if start_index == 0:
                writer.writerow([
                    "raw_index",
                    "user_id",
                    "history",
                    "recommended_item",
                    "explanation",
                    "quality_score",
                    "source_overlap",
                    "repeat_ratio",
                    "explanation_tokens"
                ])

            for raw_idx in tqdm(range(start_index, len(df)), initial=start_index, total=len(df), desc=f"Generating for {split}"):
                row = df.iloc[raw_idx]
                history = decode_history(str(row["history"]))
                target = decode_history(str(row["target"]))
                user_id = row.get('user_id', 1)
                
                prompt_text = EXPLANATION_PROMPT_TEMPLATE.format(history=history, item=target)
                explanation_text, metrics = generate_explanation(prompt_text, history, target)
                score, overlap_ratio, repeat_ratio, token_len = metrics

                should_write = True
                if "Error:" in explanation_text or explanation_text.strip() == "":
                    should_write = False
                if should_write and not is_good_explanation(score, overlap_ratio, repeat_ratio, token_len):
                    should_write = False

                if should_write:
                    writer.writerow([
                        raw_idx,
                        user_id,
                        history,
                        target,
                        explanation_text,
                        round(score, 4),
                        round(overlap_ratio, 4),
                        round(repeat_ratio, 4),
                        token_len
                    ])
                    split_kept += 1
                else:
                    split_skipped += 1

                if raw_idx % PROGRESS_SAVE_EVERY == 0:
                    with open(progress_path, "w", encoding="utf-8") as pf:
                        json.dump({"next_raw_index": raw_idx + 1}, pf)

            with open(progress_path, "w", encoding="utf-8") as pf:
                json.dump({"next_raw_index": len(df)}, pf)

        print(f"\n{split} set finished. Final dataset saved to: {output_path}")
        print(f"Kept samples: {split_kept} | Filtered samples: {split_skipped}")
        global_kept += split_kept
        global_skipped += split_skipped

    print("\n=== All dataset generation finished ===")
    print(f"Total kept samples: {global_kept}")
    print(f"Total filtered samples: {global_skipped}")

if __name__ == "__main__":
    main()
