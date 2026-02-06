import os
import pandas as pd
import torch
import evaluate 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================================================================
# 1. 配置部分 (CONFIGURATION)
# ==============================================================================

# 【修复】明确设备选择，支持CPU模式
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"

# --- 【核心】模型路径指向原始的、未经微调的BART-Base (最内层) ---
BART_BASELINE_PATH = os.path.join(PROJECT_ROOT, "models", "bart-base", "facebook", "bart-base")

# --- BERTScore依赖的roberta模型的本地路径 ---
#ROBERTA_LOCAL_PATH = os.path.join(PROJECT_ROOT, "models", "roberta-large")

# --- 评估脚本的本地路径 ---
METRICS_DIR = os.path.join(PROJECT_ROOT, "offline_metrics")
ROUGE_SCRIPT_PATH = os.path.join(METRICS_DIR, "rouge")
#BERTSCORE_SCRIPT_PATH = os.path.join(METRICS_DIR, "bertscore")

# --- 数据和结果路径 ---
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "explanation_dataset_test.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "evaluation_results_bart_baseline.csv")

# ==============================================================================
# 2. 模型加载与生成函数
# ==============================================================================

def load_bart_baseline(model_path):
    """专门加载原始的、未经微调的BART-Base模型"""
    print(f"--- 正在加载BART-Base (Zero-Shot)基线模型 ---")
    print(f"  - 路径: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        model = model.to(DEVICE)
        model.eval()
        print("BART-Base基线模型加载成功。")
        return model, tokenizer
    except Exception as e:
        print(f"!!! 加载BART-Base基线失败: {e}")
        return None, None

def generate_explanation(model, tokenizer, history, item):
    """为模型生成解释，使用与训练时一致的Prompt格式"""
    # 【修复】使用简化格式
    prompt = f"User History: {history}\nRecommended Item: {item}\nExplanation:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=768, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150, num_beams=5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================================================================
# 3. 主程序 (MAIN LOGIC)
# ==============================================================================

def main():
    bart_baseline_model, bart_baseline_tokenizer = load_bart_baseline(BART_BASELINE_PATH)
    if not bart_baseline_model:
        return
        
    print("\n--- 正在从本地加载评估指标 ---")
    try:
        rouge = evaluate.load(ROUGE_SCRIPT_PATH)
        #bertscore = evaluate.load(BERTSCORE_SCRIPT_PATH)
        print("评估指标加载成功。")
    except Exception as e:
        print(f"!!! 加载离线评估指标失败: {e}")
        return
    
    print(f"\n--- 正在加载测试数据 ---")
    try:
        df = pd.read_csv(TEST_DATA_PATH)
        # 正式评估，处理全部数据
        df = df.head(5000) 
    except FileNotFoundError:
        print(f"!!! 错误: 测试集文件未找到! '{TEST_DATA_PATH}'")
        return

    results = []
    print(f"\n--- 正在为BART-Base (Zero-Shot)基线生成 {len(df)} 条解释 ---")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        history, item, reference = str(row['history']), str(row['recommended_item']), str(row['explanation'])
        bart_pred = generate_explanation(bart_baseline_model, bart_baseline_tokenizer, history, item)
        results.append({
            'golden_explanation': reference,
            'prediction': bart_pred
        })

    results_df = pd.DataFrame(results)
    references = results_df['golden_explanation'].tolist()
    predictions = results_df['prediction'].tolist()
    
    print("\n--- 正在计算自动化评估指标 ---")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    
    # --- 【核心修复】在这里，我们明确地告诉bertscore去哪里找roberta-large ---
    #print("正在计算BERTScore (这可能需要几分钟)...")
    #bert_scores = bertscore.compute(
        #predictions=predictions, 
        #references=references, 
        #lang="en",
        #model_type="roberta-large"
    #)
    #bert_f1 = sum(bert_scores['f1'])/len(bert_scores['f1']) if bert_scores['f1'] else 0.0

    print("\n--- BART-Base (Zero-Shot)基线评估结果 ---")
    print(f"{'Metric':<15} | {'Score':<10}")
    print("-" * 30)
    print(f"{'ROUGE-1':<15} | {rouge_scores.get('rouge1', 0.0):.4f}")
    print(f"{'ROUGE-2':<15} | {rouge_scores.get('rouge2', 0.0):.4f}")
    print(f"{'ROUGE-L':<15} | {rouge_scores.get('rougeL', 0.0):.4f}")
    #print(f"{'BERTScore-F1':<15} | {bert_f1:.4f}")
    
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n详细生成结果已保存到: {RESULTS_PATH}")
    print("\n🎉 BART-Base基线评估完成！")

if __name__ == "__main__":
    main()