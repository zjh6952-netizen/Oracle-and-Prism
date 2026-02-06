#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""
快速BERTScore测试脚本
直接从CSV文件读取已有的预测结果，计算BERTScore分数
避免重新加载大模型，节省内存
"""

import os
import sys
import pandas as pd
import evaluate

# 设置路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CSV_FILE = os.path.join(RESULTS_DIR, 'evaluation_results_our_model.csv')
BERTSCORE_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'offline_metrics', 'bertscore')

# 设置评估数量（快速测试）
NUM_SAMPLES = 100

def main():
    print("="*60)
    print("BERTScore 快速测试")
    print("="*60)
    
    # 1. 检查CSV文件是否存在
    if not os.path.exists(CSV_FILE):
        print(f"❌ 错误：找不到CSV文件: {CSV_FILE}")
        sys.exit(1)
    
    print(f"✓ 找到CSV文件: {CSV_FILE}")
    
    # 2. 读取CSV文件
    print(f"\n正在读取CSV文件...")
    df = pd.read_csv(CSV_FILE)
    print(f"✓ CSV文件共有 {len(df)} 条记录")
    
    # 3. 限制评估数量（快速测试）
    df = df.head(NUM_SAMPLES)
    print(f"✓ 将评估前 {NUM_SAMPLES} 条记录进行快速测试")
    
    # 4. 提取预测和参考答案
    predictions = df['prediction'].fillna('').tolist()
    references = df['golden'].fillna('').tolist()
    
    # 统计一下数据情况
    valid_pairs = sum(1 for p, r in zip(predictions, references) if p.strip() and r.strip())
    print(f"✓ 其中有效的预测-参考对: {valid_pairs}/{len(predictions)}")
    
    # 5. 加载BERTScore评估器
    print(f"\n正在加载BERTScore评估器...")
    print(f"  - 使用离线脚本路径: {BERTSCORE_SCRIPT_PATH}")
    
    try:
        bertscore = evaluate.load(BERTSCORE_SCRIPT_PATH)
        print("✓ BERTScore评估器加载成功")
    except Exception as e:
        print(f"❌ 加载BERTScore失败: {e}")
        sys.exit(1)
    
    # 6. 计算BERTScore
    print(f"\n正在计算BERTScore...")
    print(f"  - 这可能需要几分钟时间，请耐心等待...")
    
    try:
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang='en',
            device='cpu',  # 使用CPU
            verbose=True
        )
        print("✓ BERTScore计算完成")
    except Exception as e:
        print(f"❌ 计算BERTScore失败: {e}")
        sys.exit(1)
    
    # 7. 显示结果
    print("\n" + "="*60)
    print("BERTScore 评估结果")
    print("="*60)
    
    # BERTScore返回P, R, F1三个分数
    if 'precision' in results:
        avg_precision = sum(results['precision']) / len(results['precision'])
        print(f"Precision: {avg_precision:.4f}")
    
    if 'recall' in results:
        avg_recall = sum(results['recall']) / len(results['recall'])
        print(f"Recall:    {avg_recall:.4f}")
    
    if 'f1' in results:
        avg_f1 = sum(results['f1']) / len(results['f1'])
        print(f"F1 Score:  {avg_f1:.4f}")
    
    print("="*60)
    print(f"\n✓ 快速测试完成！共评估了 {NUM_SAMPLES} 条数据")
    print(f"✓ 如需评估全部 {len(pd.read_csv(CSV_FILE))} 条数据，请修改脚本中的 NUM_SAMPLES 变量")
    
if __name__ == '__main__':
    main()
