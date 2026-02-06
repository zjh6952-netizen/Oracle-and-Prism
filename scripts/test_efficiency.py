import os
import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer

# ==============================================================================
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"

YOUR_MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "results", "20250918_165141", "best_model.mdl") 
YOUR_MODEL_BASE_PATH = os.path.join(PROJECT_ROOT, "models", "bart-base", "facebook", "bart-base")
BASELINE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "flan-t5-xxl")

DUMMY_HISTORY = "Inception, Interstellar, The Dark Knight"
DUMMY_ITEM = "The Prestige"
DUMMY_PROMPT = f"Instruction: Generate a personalized explanation for the given recommendation.\nInput: User History: {DUMMY_HISTORY}\nRecommended Item: {DUMMY_ITEM}"

WARMUP_RUNS = 10
TIMING_RUNS = 100

# ==============================================================================
# ==============================================================================

def benchmark_model(model_name, model, tokenizer):
    """
    Generic helper to warm up a model, measure GPU memory, and benchmark generation latency.
    """
    print(f"\n--- Benchmarking model: {model_name} ---")
    
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(DEVICE)
    print(f"Allocated GPU memory after load: {initial_memory / 1024**3:.2f} GB")
    
    inputs = tokenizer(DUMMY_PROMPT, return_tensors="pt").to(DEVICE)

    print(f"Running {WARMUP_RUNS} warmup iterations...")
    for _ in range(WARMUP_RUNS):
        _ = model.generate(**inputs, max_new_tokens=150, num_beams=5)
    print("Warmup complete.")

    print(f"Running {TIMING_RUNS} timed generation iterations...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(TIMING_RUNS):
        _ = model.generate(**inputs, max_new_tokens=150, num_beams=5)
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency_ms = (total_time / TIMING_RUNS) * 1000
    
    peak_memory_gb = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
    
    print("\n--- Efficiency Benchmark Result ---")
    print(f"Model: {model_name}")
    print(f"Average Latency: {avg_latency_ms:.2f} ms/explanation")
    print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB")
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return avg_latency_ms, peak_memory_gb

# ==============================================================================
# ==============================================================================

if __name__ == "__main__":
    print("--- Starting model efficiency benchmark script ---")
    results = {}

    try:
        tokenizer_ours = AutoTokenizer.from_pretrained(YOUR_MODEL_BASE_PATH, local_files_only=True)
        model_ours = AutoModelForSeq2SeqLM.from_pretrained(YOUR_MODEL_BASE_PATH, local_files_only=True)
        model_ours.load_state_dict(torch.load(YOUR_MODEL_WEIGHTS_PATH, map_location=DEVICE), strict=False)
        model_ours = model_ours.to(DEVICE)
        model_ours.eval()
        results['Primary-Model'] = benchmark_model("Primary Model", model_ours, tokenizer_ours)
    except Exception as e:
        print(f"!!! Failed while benchmarking your model: {e}")

    try:
        tokenizer_baseline = T5Tokenizer.from_pretrained(BASELINE_MODEL_PATH, local_files_only=True)
        model_baseline = T5ForConditionalGeneration.from_pretrained(
            BASELINE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        model_baseline.eval()
        results['FLAN-T5-XXL'] = benchmark_model("FLAN-T5-XXL (Baseline)", model_baseline, tokenizer_baseline)
    except Exception as e:
        print(f"!!! Failed while benchmarking baseline model: {e}")

    print("\n\n--- Final Efficiency Comparison Report ---")
    print(f"{'Model':<25} | {'Params':<10} | {'Avg. Latency (ms)':<20} | {'Peak GPU Memory (GB)':<20}")
    print("-" * 85)
    
    if 'Primary-Model' in results:
        lat, mem = results['Primary-Model']
        print(f"{'Primary Model':<25} | {'140M':<10} | {lat:<20.2f} | {mem:<20.2f}")
    
    if 'FLAN-T5-XXL' in results:
        lat, mem = results['FLAN-T5-XXL']
        print(f"{'FLAN-T5-XXL (Baseline)':<25} | {'11B':<10} | {lat:<20.2f} | {mem:<20.2f}")

    print("\nEfficiency benchmark complete.")
