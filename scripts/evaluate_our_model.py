import os
import sys
from argparse import Namespace

import evaluate
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


# ==============================================================================
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"

YOUR_MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "results", "20250918_165141", "best_model.mdl")
YOUR_MODEL_BASE_PATH = os.path.join(PROJECT_ROOT, "models", "bart-base", "facebook", "bart-base")

METRICS_DIR = os.path.join(PROJECT_ROOT, "offline_metrics")
ROUGE_SCRIPT_PATH = os.path.join(METRICS_DIR, "rouge")
BERTSCORE_SCRIPT_PATH = os.path.join(METRICS_DIR, "bertscore")

TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "explanation_dataset_test.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "evaluation_results_our_model.csv")
HUMAN_EVAL_PATH = os.path.join(PROJECT_ROOT, "results", "human_evaluation_data_for_our_model.csv")

MAX_LENGTH = 768
MAX_OUTPUT_LENGTH = 150
EVAL_BATCH_SIZE = 16
EVAL_NUM_BEAMS = 5
EVAL_REPETITION_PENALTY = 1.2
EVAL_NO_REPEAT_NGRAM_SIZE = 3
EVAL_LENGTH_PENALTY = 1.0
EVAL_MIN_NEW_TOKENS = 0
LOCAL_FILES_ONLY = True


def _prepare_imports():
    """Import project modules from GenRec, preserving your current path layout."""
    genrec_root = os.path.join(PROJECT_ROOT, "GenRec")
    if genrec_root not in sys.path:
        sys.path.insert(0, genrec_root)
    from genrec.data import Dataset  # pylint: disable=import-outside-toplevel
    from genrec.model import GenerativeModel  # pylint: disable=import-outside-toplevel
    from genrec.utils import move_to_cuda  # pylint: disable=import-outside-toplevel
    return Dataset, GenerativeModel, move_to_cuda


def load_your_bart_model(base_path, weights_path):
    """
    Load the customized GenerativeModel (internally your modified BartForConditionalGeneration).
    strict=True enforces exact weight-structure matching and avoids silent mismatch.
    """
    print("--- Loading your finetuned custom BART model ---")
    print(f"  - Base architecture: {base_path}")
    print(f"  - Finetuned weights: {weights_path}")
    try:
        Dataset, GenerativeModel, move_to_cuda = _prepare_imports()
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=LOCAL_FILES_ONLY, add_prefix_space=True)
        tokenizer.add_tokens(["<mask>"])
        config = Namespace(
            model_name=base_path,
            local_files_only=LOCAL_FILES_ONLY,
            label_smoothing=0.1,
        )
        model = GenerativeModel(config, tokenizer)
        state_dict = torch.load(weights_path, map_location=DEVICE)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        if missing_keys or unexpected_keys:
            raise RuntimeError(f"Weight mismatch: missing={missing_keys}, unexpected={unexpected_keys}")
        model = model.to(DEVICE)
        model.eval()
        print("Your model (custom architecture) loaded successfully.")
        return model, tokenizer, Dataset, move_to_cuda
    except Exception as e:
        print("!!! Failed to load your model. Ensure paths are correct and training is completed.")
        print(f"Error: {e}")
        return None, None, None, None


def main():
    your_model, your_tokenizer, Dataset, move_to_cuda = load_your_bart_model(
        YOUR_MODEL_BASE_PATH, YOUR_MODEL_WEIGHTS_PATH
    )
    if not your_model:
        return

    print("\n--- Loading evaluation metrics from local files ---")
    try:
        rouge = evaluate.load(ROUGE_SCRIPT_PATH)
        print("ROUGE metric loaded successfully.")
    except Exception as e:
        print(f"!!! Failed to load ROUGE: {e}")
        return

    try:
        bertscore = evaluate.load(BERTSCORE_SCRIPT_PATH)
        print("BERTScore metric loaded successfully.")
    except Exception as e:
        print(f"!!! Failed to load BERTScore: {e}")
        print("Evaluation will continue with ROUGE only.")
        bertscore = None

    print("\n--- Loading test data and building evaluation set ---")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"!!! Error: test dataset not found: '{TEST_DATA_PATH}'")
        return

    test_set = Dataset(
        your_tokenizer,
        max_length=MAX_LENGTH,
        path=TEST_DATA_PATH,
        max_output_length=MAX_OUTPUT_LENGTH,
        filter_pseudo_labels=False,
    )
    if len(test_set) == 0:
        print("!!! Test set is empty; evaluation cannot proceed.")
        return
    test_loader = DataLoader(
        test_set,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=test_set.collate_fn,
    )
    print(f"Number of test samples: {len(test_set)}")

    results = []
    print(f"\n--- Generating {len(test_set)} explanations with your model ---")
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            gpu_batch = move_to_cuda(batch) if DEVICE == "cuda" else batch
            predictions = your_model.predict(
                gpu_batch,
                num_beams=EVAL_NUM_BEAMS,
                max_length=MAX_OUTPUT_LENGTH,
                repetition_penalty=EVAL_REPETITION_PENALTY,
                no_repeat_ngram_size=EVAL_NO_REPEAT_NGRAM_SIZE,
                length_penalty=EVAL_LENGTH_PENALTY,
                min_new_tokens=EVAL_MIN_NEW_TOKENS,
            )
            for src, refs, pred in zip(batch.input_text, batch.target_text, predictions):
                ref = refs[0] if refs else ""
                item = ""
                if "Recommended Item:" in src:
                    item = src.split("Recommended Item:", 1)[1].split("\n", 1)[0].strip()
                results.append(
                    {
                        "history": src,
                        "item": item,
                        "golden": ref,
                        "prediction": pred,
                    }
                )

    results_df = pd.DataFrame(results)
    references = results_df["golden"].tolist()
    predictions = results_df["prediction"].tolist()

    print("\n--- Computing automatic evaluation metrics ---")
    print("Computing ROUGE...")
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    bert_scores = None
    if bertscore:
        print("Computing BERTScore (this may take a few minutes)...")
        try:
            bert_scores = bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
                device=DEVICE,
            )
            print("BERTScore computation finished")
        except Exception as e:
            print(f"!!! BERTScore computation failed: {e}")
            bert_scores = None

    print("\n--- Evaluation Results: Your Model ---")
    print(f"{'Metric':<15} | {'Score':<10}")
    print("-" * 30)
    print(f"{'ROUGE-1':<15} | {rouge_scores.get('rouge1', 0.0):.4f}")
    print(f"{'ROUGE-2':<15} | {rouge_scores.get('rouge2', 0.0):.4f}")
    print(f"{'ROUGE-L':<15} | {rouge_scores.get('rougeL', 0.0):.4f}")
    if bert_scores:
        bert_f1 = sum(bert_scores["f1"]) / len(bert_scores["f1"])
        print(f"{'BERTScore-F1':<15} | {bert_f1:.4f}")

    results_df.to_csv(RESULTS_PATH, index=False)
    results_df.to_csv(HUMAN_EVAL_PATH, index=False)
    print(f"\nDetailed generation results saved to: {RESULTS_PATH}")
    print("\nYour model evaluation is complete.")


if __name__ == "__main__":
    main()
