import pandas as pd
import random
import os

# --- 配置 ---
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"

# 输入文件路径
OUR_MODEL_RESULTS_FILE = os.path.join(PROJECT_ROOT, "results", "evaluation_results_our_model.csv")
TEACHER_BASELINE_FILE = os.path.join(PROJECT_ROOT, "results", "evaluation_results_baseline.csv")
BART_BASELINE_FILE = os.path.join(PROJECT_ROOT, "results", "evaluation_results_bart_baseline.csv")

# 输出文件路径
FINAL_SURVEY_FILE = os.path.join(PROJECT_ROOT, "results", "human_evaluation_survey_data.csv")
FINAL_SURVEY_FILE_WITH_IDS = os.path.join(PROJECT_ROOT, "results", "human_evaluation_survey_data_WITH_IDS.csv") # 新增：带模型ID的文件

# 我们随机抽取100个样本进行人工评估
NUM_SAMPLES = 100
RANDOM_STATE = 42

# --- 主逻辑 ---
print("--- 正在准备人工评估数据 ---")

# 1. 加载三个结果文件
df_ours = pd.read_csv(OUR_MODEL_RESULTS_FILE)
df_teacher = pd.read_csv(TEACHER_BASELINE_FILE)
df_bart = pd.read_csv(BART_BASELINE_FILE)

# 2. 合并成一个大的DataFrame
df_merged = pd.merge(df_ours, df_teacher, on=['history', 'item'], suffixes=('_ours', '_teacher'))
df_merged = pd.merge(df_merged, df_bart, on=['history', 'item'])
df_merged.rename(columns={'prediction': 'prediction_bart'}, inplace=True)

# 3. 随机抽样
if len(df_merged) >= NUM_SAMPLES:
    df_sample = df_merged.sample(n=NUM_SAMPLES, random_state=RANDOM_STATE)
else:
    df_sample = df_merged
print(f"已随机抽取 {len(df_sample)} 个样本用于评估。")

# 4. 【关键】匿名化和随机化
survey_data = []
survey_data_with_ids = [] # 新增：用于保存带模型ID的数据，仅供你自己分析用
model_names = ['GenRec-E', 'FLAN-T5-XXL', 'BART-Base']

for index, row in df_sample.iterrows():
    # 创建一个包含解释和其来源的列表
    explanations_with_source = [
        (row['prediction_ours'], 'GenRec-E'),
        (row['prediction_teacher'], 'FLAN-T5-XXL'),
        (row['prediction_bart'], 'BART-Base')
    ]
    
    # 随机打乱这个列表
    random.shuffle(explanations_with_source)
    
    # 构建用于问卷的匿名数据
    survey_data.append({
        'sample_id': index, # 新增：样本ID，便于后续追踪
        'history': row['history'],
        'item': row['item'],
        'explanation_A': explanations_with_source[0][0],
        'explanation_B': explanations_with_source[1][0],
        'explanation_C': explanations_with_source[2][0]
    })
    
    # 构建用于自己分析的“答案密钥”数据
    survey_data_with_ids.append({
        'sample_id': index,
        'history': row['history'],
        'item': row['item'],
        'explanation_A_text': explanations_with_source[0][0],
        'explanation_A_model': explanations_with_source[0][1],
        'explanation_B_text': explanations_with_source[1][0],
        'explanation_B_model': explanations_with_source[1][1],
        'explanation_C_text': explanations_with_source[2][0],
        'explanation_C_model': explanations_with_source[2][1],
    })
    
survey_df = pd.DataFrame(survey_data)
survey_df_with_ids = pd.DataFrame(survey_data_with_ids)

# 5. 保存最终的问卷数据
survey_df.to_csv(FINAL_SURVEY_FILE, index=False)
survey_df_with_ids.to_csv(FINAL_SURVEY_FILE_WITH_IDS, index=False) # 保存密钥
print(f"\n🎉 最终用于问卷的数据已生成！文件路径: {FINAL_SURVEY_FILE}")
print(f"🔑 用于分析结果的“答案密钥”已生成！文件路径: {FINAL_SURVEY_FILE_WITH_IDS}")
print("请将 `human_evaluation_survey_data.csv` 中的内容复制到问卷工具中。")
print("**重要：** 请妥善保管 `...WITH_IDS.csv` 文件，切勿泄露给评估员，这是你后期分析的关键！")