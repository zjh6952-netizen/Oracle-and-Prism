import os
import pandas as pd
import torch
import evaluate 
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ==============================================================================
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"

BASELINE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "flan-t5-xxl")

#ROBERTA_LOCAL_PATH = os.path.join(PROJECT_ROOT, "models", "roberta-large")

METRICS_DIR = os.path.join(PROJECT_ROOT, "offline_metrics")
ROUGE_SCRIPT_PATH = os.path.join(METRICS_DIR, "rouge")
#BERTSCORE_SCRIPT_PATH = os.path.join(METRICS_DIR, "bertscore")

TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "explanation_dataset_test.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "evaluation_results_baseline.csv")

# ==============================================================================
# ==============================================================================

def load_teacher_t5_model(model_path):
    """Load the FLAN-T5 teacher model in a high-performance baseline setup."""
    print(f"--- Loading baseline model (FLAN-T5-XXL) ---")
    print(f"  - Path: {model_path}")
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=True
        )
        model.eval()
        print("Baseline model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"!!! Failed to load baseline model: {e}"); return None, None

def generate_explanation(model, tokenizer, history, item):
    """Generate one explanation for a history-item pair."""
    prompt = f"Instruction: Generate a short and faithful explanation for the following local business recommendation.\nInput: User Visit History: {history}\nRecommended Business: {item}"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150, num_beams=5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
# ==============================================================================
# ==============================================================================

def main():
    baseline_model, baseline_tokenizer = load_teacher_t5_model(BASELINE_MODEL_PATH)
    if not baseline_model:
        return

    print("\n--- Loading evaluation metrics from local files ---")
    try:
        rouge = evaluate.load(ROUGE_SCRIPT_PATH)
        #bertscore = evaluate.load(BERTSCORE_SCRIPT_PATH)
        print("Evaluation metrics loaded successfully.")
    except Exception as e:
        print(f"!!! Failed to load offline evaluation metrics: {e}")
        return
        
    print(f"\n--- Loading test data ---")
    try:
        df = pd.read_csv(TEST_DATA_PATH)
        #df = df.head(1000)
    except FileNotFoundError:
        print(f"!!! Error: test dataset not found: '{TEST_DATA_PATH}'")
        return

    results = []
    print(f"\n--- Generating {len(df)} explanations with baseline model (FLAN-T5-XXL) ---")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        history, item, reference = str(row['history']), str(row['recommended_item']), str(row['explanation'])
        baseline_pred = generate_explanation(baseline_model, baseline_tokenizer, history, item)
        results.append({'golden': reference, 'prediction': baseline_pred})
        
    results_df = pd.DataFrame(results)
    references = results_df['golden'].tolist()
    predictions = results_df['prediction'].tolist()
    
    print("\n--- Computing automatic evaluation metrics ---")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    
   # bert_scores = bertscore.compute(
        #predictions=predictions, 
        #references=references, 
        #lang="en",
        #model_type="roberta-large"
    #)
    #bert_f1 = sum(bert_scores['f1'])/len(bert_scores['f1']) if bert_scores['f1'] else 0.0

    print("\n--- Evaluation Results: Baseline Model (FLAN-T5-XXL) ---")
    print(f"{'Metric':<15} | {'Score':<10}")
    print("-" * 30)
    print(f"{'ROUGE-1':<15} | {rouge_scores.get('rouge1', 0.0):.4f}")
    print(f"{'ROUGE-2':<15} | {rouge_scores.get('rouge2', 0.0):.4f}")
    print(f"{'ROUGE-L':<15} | {rouge_scores.get('rougeL', 0.0):.4f}")
    #print(f"{'BERTScore-F1':<15} | {bert_f1:.4f}")

    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nDetailed generation results saved to: {RESULTS_PATH}")
    print("\nBaseline model evaluation is complete.")

if __name__ == "__main__":
    main()
