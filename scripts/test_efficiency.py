import os
import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer

# ==============================================================================
# 1. 配置部分 (CONFIGURATION)
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"

# --- 模型路径 ---
# !! 关键 !!: 请确保这里的 'YYYYMMDD_HHMMSS' 是你真实的训练输出文件夹的时间戳
YOUR_MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "results", "20250918_165141", "best_model.mdl") 
YOUR_MODEL_BASE_PATH = os.path.join(PROJECT_ROOT, "models", "bart-base", "facebook", "bart-base")
BASELINE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "flan-t5-xxl")

# 我们只需要一条样例输入来进行速度测试
DUMMY_HISTORY = "Inception, Interstellar, The Dark Knight"
DUMMY_ITEM = "The Prestige"
DUMMY_PROMPT = f"Instruction: Generate a personalized explanation for the given recommendation.\nInput: User History: {DUMMY_HISTORY}\nRecommended Item: {DUMMY_ITEM}"

# 测试参数
WARMUP_RUNS = 10      # 预热次数
TIMING_RUNS = 100     # 正式计时次数

# ==============================================================================
# 2. 核心测试函数
# ==============================================================================

def benchmark_model(model_name, model, tokenizer):
    """
    一个通用的函数，用于加载模型、测量显存并测试其生成延迟。
    """
    print(f"\n--- 正在测试模型: {model_name} ---")
    
    # --- 1. 测量显存占用 ---
    # PyTorch的显存管理比较复杂，我们测量已分配的显存
    torch.cuda.empty_cache() # 清理一下缓存
    initial_memory = torch.cuda.memory_allocated(DEVICE)
    # 实际上模型已经在加载时被分配了显存，这里我们主要看峰值
    print(f"模型加载后，已分配显存: {initial_memory / 1024**3:.2f} GB")
    
    # --- 2. 编码输入 ---
    inputs = tokenizer(DUMMY_PROMPT, return_tensors="pt").to(DEVICE)

    # --- 3. GPU预热 (Warmup) ---
    print(f"正在进行 {WARMUP_RUNS} 次预热...")
    for _ in range(WARMUP_RUNS):
        _ = model.generate(**inputs, max_new_tokens=150, num_beams=5)
    print("预热完成。")

    # --- 4. 正式计时 ---
    print(f"正在进行 {TIMING_RUNS} 次计时生成...")
    torch.cuda.synchronize() # 等待所有GPU操作完成
    start_time = time.time()
    
    for _ in range(TIMING_RUNS):
        _ = model.generate(**inputs, max_new_tokens=150, num_beams=5)
        
    torch.cuda.synchronize() # 再次等待，确保所有生成都已结束
    end_time = time.time()
    
    # --- 5. 计算并报告结果 ---
    total_time = end_time - start_time
    avg_latency_ms = (total_time / TIMING_RUNS) * 1000
    
    # 记录峰值显存
    peak_memory_gb = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
    
    print("\n--- 效率测试结果 ---")
    print(f"模型: {model_name}")
    print(f"平均延迟 (Avg. Latency): {avg_latency_ms:.2f} ms/explanation")
    print(f"峰值显存占用 (Peak GPU Memory): {peak_memory_gb:.2f} GB")
    
    # 清理显存，为下一个模型做准备
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return avg_latency_ms, peak_memory_gb

# ==============================================================================
# 3. 主程序
# ==============================================================================

if __name__ == "__main__":
    print("--- 启动模型效率测试脚本 ---")
    results = {}

    # --- 测试你的模型 (GenRec-E) ---
    try:
        tokenizer_ours = AutoTokenizer.from_pretrained(YOUR_MODEL_BASE_PATH, local_files_only=True)
        model_ours = AutoModelForSeq2SeqLM.from_pretrained(YOUR_MODEL_BASE_PATH, local_files_only=True)
        model_ours.load_state_dict(torch.load(YOUR_MODEL_WEIGHTS_PATH, map_location=DEVICE), strict=False)
        model_ours = model_ours.to(DEVICE)
        model_ours.eval()
        results['GenRec-E'] = benchmark_model("GenRec-E (Ours)", model_ours, tokenizer_ours)
    except Exception as e:
        print(f"!!! 测试你的模型时失败: {e}")

    # --- 测试基线模型 (FLAN-T5-XXL) ---
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
        print(f"!!! 测试基线模型时失败: {e}")

    # --- 打印最终的对比表格 ---
    print("\n\n--- 最终效率对比报告 ---")
    print(f"{'Model':<25} | {'Params':<10} | {'Avg. Latency (ms)':<20} | {'Peak GPU Memory (GB)':<20}")
    print("-" * 85)
    
    if 'GenRec-E' in results:
        lat, mem = results['GenRec-E']
        print(f"{'GenRec-E (Ours)':<25} | {'140M':<10} | {lat:<20.2f} | {mem:<20.2f}")
    
    if 'FLAN-T5-XXL' in results:
        lat, mem = results['FLAN-T5-XXL']
        print(f"{'FLAN-T5-XXL (Baseline)':<25} | {'11B':<10} | {lat:<20.2f} | {mem:<20.2f}")

    print("\n🎉 效率测试完成！")