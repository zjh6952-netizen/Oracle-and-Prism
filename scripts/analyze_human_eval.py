import pandas as pd
import numpy as np
from scipy import stats
import os
import re

PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
RAW_RESULTS_FILE = os.path.join(PROJECT_ROOT, "results", "raw_survey_results_exported_from_wjx.csv")
KEY_FILE = os.path.join(PROJECT_ROOT, "results", "human_evaluation_survey_data_WITH_IDS.csv")
STATS_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "final_human_eval_stats.csv")
PLOT_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "results", "human_eval_plot.png")

print("--- Starting human evaluation analysis ---")

print("1. Loading data...")
raw_df = pd.read_csv(RAW_RESULTS_FILE)
key_df = pd.read_csv(KEY_FILE)

print(f"  Raw data has {len(raw_df)} rows (evaluator submissions)")
print(f"  Raw data column examples: {list(raw_df.columns)[:5]}...")

print("2. Reshaping data (this may take a while)...")

all_scores = []

for scene_index, scene_row in key_df.iterrows():
    sample_id = scene_row['sample_id']
    
    for evaluator_index, evaluator_row in raw_df.iterrows():
        base_pattern = f"场景{sample_id+1}|场景{scene_index+1}"
        
        for expl_char in ['A', 'B', 'C']:
            target_model = scene_row[f'explanation_{expl_char}_model']
            target_text = scene_row[f'explanation_{expl_char}_text']
            
            for metric in ['说服力', '个性化', '忠实度']:
                pattern = f".*{base_pattern}.*解释{expl_char}.*{metric}.*"
                
                matching_columns = [col for col in raw_df.columns if re.search(pattern, col, re.IGNORECASE)]
                
                if not matching_columns:
                    print(f"  Warning: no matching column found: {pattern}")
                    continue
                if len(matching_columns) > 1:
                    print(f"  Warning: multiple matching columns found {matching_columns}; using the first one")
                
                target_column = matching_columns[0]
                score_value = evaluator_row[target_column]
                
                all_scores.append({
                    'sample_id': sample_id,
                    'evaluator_id': evaluator_index,
                    'model': target_model,
                    'metric': metric,
                    'score': score_value,
                    'explanation_text': target_text
                })

scores_df = pd.DataFrame(all_scores)
print(f"  Successfully collected {len(scores_df)} score records.")

print("3. Computing descriptive statistics...")
stats_df = scores_df.groupby(['model', 'metric']).agg(
    mean_score=('score', 'mean'),
    std_score=('score', 'std'),
    count=('score', 'count')
).reset_index()

print("\n--- Final Statistics (Mean ± Std) ---")
for model in ['Primary-Model', 'FLAN-T5-XXL', 'BART-Base']:
    print(f"\n{model}:")
    model_data = stats_df[stats_df['model'] == model]
    for _, row in model_data.iterrows():
        print(f"  {row['metric']}: {row['mean_score']:.3f} ± {row['std_score']:.3f}")

print("\n4. Running significance tests (Paired T-Test)...")
pivot_df = scores_df.pivot_table(
    index=['sample_id', 'evaluator_id', 'metric'],
    columns='model',
    values='score'
).reset_index()

p_values = {}

baselines = ['FLAN-T5-XXL', 'BART-Base']
metrics = ['说服力', '个性化', '忠实度']

for baseline in baselines:
    for metric in metrics:
        data_primary = pivot_df[pivot_df['metric'] == metric]['Primary-Model'].dropna()
        data_baseline = pivot_df[pivot_df['metric'] == metric][baseline].dropna()
        
        min_len = min(len(data_primary), len(data_baseline))
        data_primary = data_primary.iloc[:min_len]
        data_baseline = data_baseline.iloc[:min_len]
        
        t_stat, p_val = stats.ttest_rel(data_primary, data_baseline)
        p_values[f'Primary-Model_vs_{baseline}_{metric}'] = p_val
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"Primary-Model vs {baseline} ({metric}): p = {p_val:.5f} {significance}")

print("5. Saving results...")
stats_df.to_csv(STATS_OUTPUT_FILE, index=False, encoding='utf-8-sig')

p_df = pd.DataFrame(list(p_values.items()), columns=['comparison', 'p_value'])
p_df.to_csv(STATS_OUTPUT_FILE.replace('.csv', '_pvalues.csv'), index=False, encoding='utf-8-sig')

print(f"\nAnalysis completed.")
print(f"  Descriptive statistics saved to: {STATS_OUTPUT_FILE}")
print(f"  Significance test results saved to: {STATS_OUTPUT_FILE.replace('.csv', '_pvalues.csv')}")
