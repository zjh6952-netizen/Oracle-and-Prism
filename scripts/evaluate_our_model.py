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
# 1. 配置部分 (CONFIGURATION)
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"

# !! 关键 !!: 请将 'YYYYMMDD_HHMMSS' 替换为你真实的训练输出文件夹的时间戳
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
    加载自定义 GenerativeModel（内部是你改造过的 BartForConditionalGeneration）。
    strict=True 保证权重结构必须匹配，避免 silently ignore。
    """
    print("--- 正在加载你微调好的自定义BART模型 ---")
    print(f"  - 基础结构: {base_path}")
    print(f"  - 微调权重: {weights_path}")
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
            raise RuntimeError(f"权重不匹配: missing={missing_keys}, unexpected={unexpected_keys}")
        model = model.to(DEVICE)
        model.eval()
        print("你的模型(GenRec-E, 自定义结构)加载成功。")
        return model, tokenizer, Dataset, move_to_cuda
    except Exception as e:
        print("!!! 加载你的模型失败! 请确保路径正确且训练已完成。")
        print(f"错误: {e}")
        return None, None, None, None


def main():
    your_model, your_tokenizer, Dataset, move_to_cuda = load_your_bart_model(
        YOUR_MODEL_BASE_PATH, YOUR_MODEL_WEIGHTS_PATH
    )
    if not your_model:
        return

    print("\n--- 正在从本地加载评估指标 ---")
    try:
        rouge = evaluate.load(ROUGE_SCRIPT_PATH)
        print("✓ ROUGE评估指标加载成功。")
    except Exception as e:
        print(f"!!! 加载ROUGE失败: {e}")
        return

    try:
        bertscore = evaluate.load(BERTSCORE_SCRIPT_PATH)
        print("✓ BERTScore评估指标加载成功。")
    except Exception as e:
        print(f"!!! 加载BERTScore失败: {e}")
        print("将仅使用ROUGE进行评估")
        bertscore = None

    print("\n--- 正在加载测试数据并构建评估集 ---")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"!!! 错误: 测试集文件未找到! '{TEST_DATA_PATH}'")
        return

    test_set = Dataset(
        your_tokenizer,
        max_length=MAX_LENGTH,
        path=TEST_DATA_PATH,
        max_output_length=MAX_OUTPUT_LENGTH,
        filter_pseudo_labels=False,
    )
    if len(test_set) == 0:
        print("!!! 测试集为空，无法评估。")
        return
    test_loader = DataLoader(
        test_set,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=test_set.collate_fn,
    )
    print(f"测试样本数: {len(test_set)}")

    results = []
    print(f"\n--- 正在为你的模型 (GenRec-E) 生成 {len(test_set)} 条解释 ---")
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

    print("\n--- 正在计算自动化评估指标 ---")
    print("正在计算ROUGE...")
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    bert_scores = None
    if bertscore:
        print("正在计算BERTScore (可能需要几分钟，请耐心等待)...")
        try:
            bert_scores = bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
                device=DEVICE,
            )
            print("✓ BERTScore计算完成")
        except Exception as e:
            print(f"!!! BERTScore计算失败: {e}")
            bert_scores = None

    print("\n--- 你的模型 (GenRec-E) 评估结果 ---")
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
    print(f"\n详细生成结果已保存到: {RESULTS_PATH}")
    print("\n🎉 你的模型评估完成！")


if __name__ == "__main__":
    main()
