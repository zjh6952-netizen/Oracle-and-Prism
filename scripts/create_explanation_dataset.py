import os
import time
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import csv # 使用更底层的csv库来实现内存高效的追加写入

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

def generate_explanation(prompt):
    """使用加载好的BF16模型生成解释，并加入了安全截断。"""
    try:
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        ).to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=150, no_repeat_ngram_size=2)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"\n!!! 模型推理时发生错误: {e}")
        return "Error: Generation failed."

# ==============================================================================
# 3. 主程序 (MAIN LOGIC) - 内存优化版
# ==============================================================================

def main():
    # 循环处理训练集和测试集
    for split, filepath in INPUT_FILES.items():
        print(f"\n--- 开始创建解释数据集: {split} set (内存优化模式) ---")
        output_path = os.path.join(OUTPUT_DIR, f"explanation_dataset_{split}.csv")
        
        try:
            df = pd.read_csv(filepath)
            print(f"成功读取 {len(df)} 条原始数据。")
        except FileNotFoundError:
            print(f"!!! 错误: 原始数据文件未找到! 请检查路径 '{filepath}'")
            continue

        start_index = 0
        # --- 核心修改：不再把旧文件读入内存，只用它来确定起始位置 ---
        if os.path.exists(output_path):
            try:
                # 只读取一小部分来获取行数，避免加载整个文件
                processed_df_len = pd.read_csv(output_path, usecols=[0]).shape[0]
                start_index = processed_df_len
                print(f"发现已存在的输出文件，其中包含 {start_index} 条数据。将从该位置开始追加。")
            except (pd.errors.EmptyDataError, FileNotFoundError):
                start_index = 0
                print("发现空的或损坏的输出文件，将从头开始创建。")
        
        # --- 核心修改：使用'a' (append)模式和csv库进行流式写入 ---
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 如果是新文件（或空文件），就先写入表头
            if start_index == 0:
                writer.writerow(["user_id", "history", "recommended_item", "explanation"])

            # 循环处理剩余的数据
            for index, row in tqdm(df.iloc[start_index:].iterrows(), initial=start_index, total=len(df), desc=f"Generating for {split}"):
                history = str(row['history'])
                target = str(row['target'])
                user_id = row.get('user_id', 1)
                
                prompt_text = EXPLANATION_PROMPT_TEMPLATE.format(history=history, item_to_explain=target)
                explanation_text = generate_explanation(prompt_text)
                
                # --- 核心修改：生成一条，就立刻写入磁盘 ---
                if "Error:" not in explanation_text and explanation_text.strip() != "":
                    writer.writerow([user_id, history, target, explanation_text])

        print(f"\n🎉 {split} set 处理完毕！最终数据集已完整保存到: {output_path}")

if __name__ == "__main__":
    main()