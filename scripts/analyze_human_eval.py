import pandas as pd
import numpy as np
from scipy import stats
import os
import re

# --- 配置 ---
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
# 你从问卷网站导出的原始打分数据 (多位评估者的答卷)
RAW_RESULTS_FILE = os.path.join(PROJECT_ROOT, "results", "raw_survey_results_exported_from_wjx.csv")
# 我们之前生成的“答案密钥” (记录了A/B/C对应哪个模型)
KEY_FILE = os.path.join(PROJECT_ROOT, "results", "human_evaluation_survey_data_WITH_IDS.csv")
# 最终统计结果的输出路径
STATS_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "final_human_eval_stats.csv")
PLOT_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "human_eval_plot.png") # 新增：图表输出

# --- 主逻辑 ---
print("--- 开始统计人工评估结果 ---")

# 1. 加载数据
print("1. 正在加载数据...")
raw_df = pd.read_csv(RAW_RESULTS_FILE)
key_df = pd.read_csv(KEY_FILE)

# 打印原始数据的列名，方便调试
print(f"  原始数据共有 {len(raw_df)} 行（评估者答卷数）")
print(f"  原始数据列名示例: {list(raw_df.columns)[:5]}...") 

# 2. 数据整理：将“宽表”变成“长表”
print("2. 正在重塑数据（这可能需要一点时间）...")

# 初始化一个空列表来存储所有打分记录
all_scores = []

# 遍历“答案密钥”中的每一个场景（样本）
for scene_index, scene_row in key_df.iterrows():
    sample_id = scene_row['sample_id']
    
    # 遍历原始数据中的每一位评估者
    for evaluator_index, evaluator_row in raw_df.iterrows():
        # 构建我们要在原始数据中查找的列名模式
        # 例如：查找包含“场景1”、“解释A”、“说服力”的列
        base_pattern = f"场景{sample_id+1}|场景{scene_index+1}" # 问卷星可能从场景1或索引+1开始
        
        for expl_char in ['A', 'B', 'C']:
            # 从“密钥”中获取当前解释字符对应的模型
            target_model = scene_row[f'explanation_{expl_char}_model']
            target_text = scene_row[f'explanation_{expl_char}_text']
            
            # 在原始数据中查找匹配的列
            for metric in ['说服力', '个性化', '忠实度']:
                # 构建一个更灵活的模式：匹配包含（场景ID、解释字符、维度）的列
                pattern = f".*{base_pattern}.*解释{expl_char}.*{metric}.*"
                
                # 使用正则表达式查找匹配的列
                matching_columns = [col for col in raw_df.columns if re.search(pattern, col, re.IGNORECASE)]
                
                if not matching_columns:
                    print(f"  警告: 未找到匹配的列: {pattern}")
                    continue
                if len(matching_columns) > 1:
                    print(f"  警告: 找到多个匹配列 {matching_columns}，将使用第一个")
                
                target_column = matching_columns[0]
                score_value = evaluator_row[target_column]
                
                # 将这条打分记录添加到列表中
                all_scores.append({
                    'sample_id': sample_id,
                    'evaluator_id': evaluator_index,
                    'model': target_model,
                    'metric': metric,
                    'score': score_value,
                    'explanation_text': target_text # 可选，用于调试
                })

# 将列表转换为DataFrame
scores_df = pd.DataFrame(all_scores)
print(f"  成功整理出 {len(scores_df)} 条打分记录。")

# 3. 计算平均分和标准差
print("3. 正在计算描述性统计量...")
# 按模型和维度分组计算
stats_df = scores_df.groupby(['model', 'metric']).agg(
    mean_score=('score', 'mean'),
    std_score=('score', 'std'),
    count=('score', 'count')
).reset_index()

print("\n--- 最终统计结果 (平均分 ± 标准差) ---")
for model in ['GenRec-E', 'FLAN-T5-XXL', 'BART-Base']:
    print(f"\n{model}:")
    model_data = stats_df[stats_df['model'] == model]
    for _, row in model_data.iterrows():
        print(f"  {row['metric']}: {row['mean_score']:.3f} ± {row['std_score']:.3f}")

# 4. 统计显著性检验 (配对T检验)
print("\n4. 正在进行显著性检验 (Paired T-Test)...")
# 我们需要为每个样本、每个评估者、每个维度，构建一个包含三个模型分数的宽表
pivot_df = scores_df.pivot_table(
    index=['sample_id', 'evaluator_id', 'metric'],
    columns='model',
    values='score'
).reset_index()

# 初始化一个字典来存储p-value结果
p_values = {}

# 对比 GenRec-E 与每个基线模型，在每个维度上
baselines = ['FLAN-T5-XXL', 'BART-Base']
metrics = ['说服力', '个性化', '忠实度']

for baseline in baselines:
    for metric in metrics:
        # 提取当前维度的数据
        data_genrec = pivot_df[pivot_df['metric'] == metric]['GenRec-E'].dropna()
        data_baseline = pivot_df[pivot_df['metric'] == metric][baseline].dropna()
        
        # 确保数据长度一致（基于相同的样本和评估者）
        min_len = min(len(data_genrec), len(data_baseline))
        data_genrec = data_genrec.iloc[:min_len]
        data_baseline = data_baseline.iloc[:min_len]
        
        # 执行配对T检验
        t_stat, p_val = stats.ttest_rel(data_genrec, data_baseline)
        p_values[f'GenRec-E_vs_{baseline}_{metric}'] = p_val
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"GenRec-E vs {baseline} ({metric}): p = {p_val:.5f} {significance}")

# 5. 保存结果
print("5. 正在保存结果...")
# 保存统计结果
stats_df.to_csv(STATS_OUTPUT_FILE, index=False, encoding='utf-8-sig')

# 保存p-value结果
p_df = pd.DataFrame(list(p_values.items()), columns=['comparison', 'p_value'])
p_df.to_csv(STATS_OUTPUT_FILE.replace('.csv', '_pvalues.csv'), index=False, encoding='utf-8-sig')

print(f"\n🎉 分析完成！")
print(f"  描述性统计结果已保存至: {STATS_OUTPUT_FILE}")
print(f"  显著性检验结果已保存至: {STATS_OUTPUT_FILE.replace('.csv', '_pvalues.csv')}")