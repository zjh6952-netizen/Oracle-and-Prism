import os
import time
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import csv # 使用更底层的csv库来实现内存高效的追加写入
import re
import json

# ==============================================================================
# 1. 配置部分 (CONFIGURATION)
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

# 生成多个候选后做重排，显著减少教师“胡说”和“复读”
NUM_CANDIDATES = 4
SAMPLING_TOP_P = 0.9
SAMPLING_TEMPERATURE = 0.8
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 3

# 伪标签质量过滤阈值
MIN_EXPLANATION_TOKENS = 6
MAX_EXPLANATION_TOKENS = 80
MAX_REPEAT_RATIO = 0.55
MIN_SOURCE_OVERLAP = 0.12
MIN_QUALITY_SCORE = 0.35
PROGRESS_SAVE_EVERY = 100

# 这是你应该在“Yelp版本”的数据生成脚本中使用的最终Prompt

EXPLANATION_PROMPT_TEMPLATE = """Generate a short and faithful explanation for the following local business recommendation.
The explanation MUST be based ONLY on the user's visit history. Do NOT invent reasons.

Context:
- User Visit History: {history}
- Recommended Business: {item_to_explain}

Explanation:
"""

# ==============================================================================
# 2. 模型加载与推理函数 (MODEL & INFERENCE) - RTX 4090 优化
# ==============================================================================

print("--- 初始化教师模型 (RTX 4090 - BF16 高性能模式) ---")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备: {DEVICE}")

try:
    tokenizer = T5Tokenizer.from_pretrained(TEACHER_MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(
        TEACHER_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    print("教师模型以BF16高性能模式加载成功。")
except Exception as e:
    print(f"!!! 加载教师模型失败! 请检查路径 '{TEACHER_MODEL_PATH}' 是否正确且完整。")
    print(f"错误信息: {e}")
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
    质量分 = 来源重合度 + 多样性 - 复读惩罚
    目的：过滤幻觉/模板化重复解释。
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
    """使用多候选重排生成解释，降低幻觉和复读。"""
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

        # 若采样候选质量太差，用 beam-search 再兜底一次。
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
        print(f"\n!!! 模型推理时发生错误: {e}")
        return "Error: Generation failed.", (-1.0, 0.0, 1.0, 0)

# ==============================================================================
# 3. 主程序 (MAIN LOGIC) - 内存优化版
# ==============================================================================

def main():
    global_kept = 0
    global_skipped = 0

    # 循环处理训练集和测试集
    for split, filepath in INPUT_FILES.items():
        print(f"\n--- 开始创建解释数据集: {split} set (内存优化模式) ---")
        output_path = os.path.join(OUTPUT_DIR, f"explanation_dataset_{split}.csv")
        progress_path = output_path + ".progress.json"
        split_kept = 0
        split_skipped = 0
        
        try:
            df = pd.read_csv(filepath)
            print(f"成功读取 {len(df)} 条原始数据。")
        except FileNotFoundError:
            print(f"!!! 错误: 原始数据文件未找到! 请检查路径 '{filepath}'")
            continue

        start_index = 0
        if os.path.exists(progress_path):
            try:
                with open(progress_path, "r", encoding="utf-8") as pf:
                    progress_state = json.load(pf)
                start_index = int(progress_state.get("next_raw_index", 0))
                print(f"发现进度文件，将从原始行号 {start_index} 继续处理。")
            except Exception:
                start_index = 0
                print("进度文件损坏，将从头开始处理。")
        elif os.path.exists(output_path):
            # 若进度文件缺失，优先从 raw_index 恢复；没有该列时只能从头开始确保正确性。
            try:
                existing_df = pd.read_csv(output_path, usecols=["raw_index"])
                if len(existing_df) > 0:
                    start_index = int(existing_df["raw_index"].max()) + 1
                    print(f"进度文件缺失，已从 raw_index 恢复到 {start_index}。")
            except Exception:
                start_index = 0
                print("进度文件缺失且输出文件不含 raw_index，将从头重新生成以避免错位。")

        if start_index >= len(df):
            print("该 split 已全部处理完成，跳过。")
            continue
        
        open_mode = "a" if start_index > 0 else "w"
        # --- 核心修改：使用流式写入 + 可恢复进度 ---
        with open(output_path, open_mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 如果是新文件（或空文件），就先写入表头
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

            # 循环处理剩余的数据
            for raw_idx in tqdm(range(start_index, len(df)), initial=start_index, total=len(df), desc=f"Generating for {split}"):
                row = df.iloc[raw_idx]
                history = decode_history(str(row["history"]))
                target = decode_history(str(row["target"]))
                user_id = row.get('user_id', 1)
                
                prompt_text = EXPLANATION_PROMPT_TEMPLATE.format(history=history, item_to_explain=target)
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

            # 确保本 split 结束后持久化到末尾
            with open(progress_path, "w", encoding="utf-8") as pf:
                json.dump({"next_raw_index": len(df)}, pf)

        print(f"\n🎉 {split} set 处理完毕！最终数据集已完整保存到: {output_path}")
        print(f"保留样本: {split_kept} | 过滤样本: {split_skipped}")
        global_kept += split_kept
        global_skipped += split_skipped

    print("\n=== 全部数据生成完成 ===")
    print(f"总保留样本: {global_kept}")
    print(f"总过滤样本: {global_skipped}")

if __name__ == "__main__":
    main()
