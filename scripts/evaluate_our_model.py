import os
import traceback
import pandas as pd
import torch
import evaluate 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM # 注意这里加了AutoModel
# 【核心】我们直接导入bert_score的核心工具，不再依赖evaluate库来加载它
#from bert_score import BERTScorer

# ==============================================================================
# 1. 配置部分 (CONFIGURATION)
# ==============================================================================

# 【修复】明确设备选择，支持CPU模式
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"

# !! 关键 !!: 请将 'YYYYMMDD_HHMMSS' 替换为你真实的训练输出文件夹的时间戳
YOUR_MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "results", "20250918_165141", "best_model.mdl") 

# 学生模型的基础结构路径 (指向最内层)
YOUR_MODEL_BASE_PATH = os.path.join(PROJECT_ROOT, "models", "bart-base", "facebook", "bart-base")

# BERTScore依赖的roberta模型的本地路径
ROBERTA_LOCAL_PATH = os.path.join(PROJECT_ROOT, "models", "roberta-large")

# 评估脚本的本地路径
METRICS_DIR = os.path.join(PROJECT_ROOT, "offline_metrics")
ROUGE_SCRIPT_PATH = os.path.join(METRICS_DIR, "rouge")
BERTSCORE_SCRIPT_PATH = os.path.join(METRICS_DIR, "bertscore")

# 数据和结果路径
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "explanation_dataset_test.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "evaluation_results_our_model.csv")
HUMAN_EVAL_PATH = os.path.join(PROJECT_ROOT, "results", "human_evaluation_data_for_our_model.csv")

# 移除这行，因为我们不再依赖HF_HOME来识别bert_score的模型路径
# os.environ['HF_HOME'] = os.path.join(PROJECT_ROOT, "models") 
# transformers_offline=1 仍可保留，以防止意外的网络请求，但在这里并非必需，因为我们将直接传入模型
# os.environ['TRANSFORMERS_OFFLINE'] = "1" 

# ==============================================================================
# 2. 模型加载与生成函数
# ==============================================================================

def load_your_bart_model(base_path, weights_path):
    """专门加载我们微调好的BART学生模型"""
    print(f"--- 正在加载你微调好的BART模型 ---")
    print(f"  - 基础结构: {base_path}")
    print(f"  - 微调权重: {weights_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_path, local_files_only=True)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE), strict=False)
        model = model.to(DEVICE)
        model.eval()
        print("你的模型(GenRec-E)加载成功。")
        return model, tokenizer
    except Exception as e:
        print(f"!!! 加载你的模型失败! 请确保路径正确且训练已完成。"); print(f"错误: {e}"); return None, None

def generate_explanation(model, tokenizer, history, item):
    """为模型生成解释"""
    # 【修复】使用与训练时一致的简化格式
    prompt = f"User History: {history}\nRecommended Item: {item}\nExplanation:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=768, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150, num_beams=5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================================================================
# 3. 主程序 (MAIN LOGIC)
# ==============================================================================

def main():
    your_model, your_tokenizer = load_your_bart_model(YOUR_MODEL_BASE_PATH, YOUR_MODEL_WEIGHTS_PATH)
    if not your_model: 
        return
        
    print("\n--- 正在从本地加载评估指标 ---")
    try:
        rouge = evaluate.load(ROUGE_SCRIPT_PATH)
        print("✓ ROUGE评估指标加载成功。")
    except Exception as e:
        print(f"!!! 加载ROUGE失败: {e}"); return
    
    try:
        bertscore = evaluate.load(BERTSCORE_SCRIPT_PATH)
        print("✓ BERTScore评估指标加载成功。")
    except Exception as e:
        print(f"!!! 加载BERTScore失败: {e}")
        print("将仅使用ROUGE进行评估")
        bertscore = None

    # --- 核心修复：直接加载roberta-large模型和分词器，然后传给BERTScorer ---
    #print("正在预加载BERTScore所需的roberta-large模型和分词器...")
    #roberta_tokenizer = None
    #roberta_model = None
    #try:
        # 直接使用你的本地路径加载tokenizer
       # roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_LOCAL_PATH, local_files_only=True)
        # 直接使用你的本地路径加载model
        #roberta_model = AutoModel.from_pretrained(ROBERTA_LOCAL_PATH, local_files_only=True)
        #print("roberta-large模型和分词器预加载成功。")
    #except Exception as e:
        #print(f"!!! 预加载roberta-large模型或分词器失败: {e}")
        #traceback.print_exc()
        #return

    #print("正在初始化BERTScorer (将使用预加载的roberta-large)...")
    #try:
        # 将已加载的tokenizer和model直接传递给BERTScorer
        # 注意：这里不再需要model_type参数，因为它会被model和tokenizer参数覆盖
        #scorer = BERTScorer(model=roberta_model, tokenizer=roberta_tokenizer, lang="en", rescale_with_baseline=True, device=DEVICE)
        #print("BERTScorer初始化成功。")
    #except Exception as e:
       # print(f"!!! 初始化BERTScorer失败: {e}")
       # traceback.print_exc() 
        #return
    
    # ... (其余代码保持不变) ...

    print(f"\n--- 正在加载测试数据 ---")
    try:
        df = pd.read_csv(TEST_DATA_PATH)
        # 【快速测试】只评估100条进行BERTScore快速测试
        df = df.head(100)
        print(f"加载了 {len(df)} 条测试数据")
    except FileNotFoundError:
        print(f"!!! 错误: 测试集文件未找到! '{TEST_DATA_PATH}'"); return

    results = []
    print(f"\n--- 正在为你的模型 (GenRec-E) 生成 {len(df)} 条解释 ---")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        history, item, reference = str(row['history']), str(row['recommended_item']), str(row['explanation'])
        your_pred = generate_explanation(your_model, your_tokenizer, history, item)
        results.append({ 
            'history': history, 'item': item,
            'golden': reference, 'prediction': your_pred
        })
        
    results_df = pd.DataFrame(results)
    references = results_df['golden'].tolist()
    predictions = results_df['prediction'].tolist()
    
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
                device=DEVICE
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
        bert_f1 = sum(bert_scores['f1']) / len(bert_scores['f1'])
        print(f"{'BERTScore-F1':<15} | {bert_f1:.4f}")
    
    results_df.to_csv(RESULTS_PATH, index=False)
    results_df.to_csv(HUMAN_EVAL_PATH, index=False)
    print(f"\n详细生成结果已保存到: {RESULTS_PATH}")
    print("\n🎉 你的模型评估完成！")

if __name__ == "__main__":
    main()