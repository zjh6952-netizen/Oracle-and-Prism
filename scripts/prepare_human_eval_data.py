import pandas as pd
import random
import os

PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"

OUR_MODEL_RESULTS_FILE = os.path.join(PROJECT_ROOT, "results", "evaluation_results_our_model.csv")
TEACHER_BASELINE_FILE = os.path.join(PROJECT_ROOT, "results", "evaluation_results_baseline.csv")
BART_BASELINE_FILE = os.path.join(PROJECT_ROOT, "results", "evaluation_results_bart_baseline.csv")

FINAL_SURVEY_FILE = os.path.join(PROJECT_ROOT, "results", "human_evaluation_survey_data.csv")
FINAL_SURVEY_FILE_WITH_IDS = os.path.join(PROJECT_ROOT, "results", "human_evaluation_survey_data_WITH_IDS.csv")

NUM_SAMPLES = 100
RANDOM_STATE = 42

print("--- Preparing human evaluation data ---")

df_ours = pd.read_csv(OUR_MODEL_RESULTS_FILE)
df_teacher = pd.read_csv(TEACHER_BASELINE_FILE)
df_bart = pd.read_csv(BART_BASELINE_FILE)

df_merged = pd.merge(df_ours, df_teacher, on=['history', 'item'], suffixes=('_ours', '_teacher'))
df_merged = pd.merge(df_merged, df_bart, on=['history', 'item'])
df_merged.rename(columns={'prediction': 'prediction_bart'}, inplace=True)

if len(df_merged) >= NUM_SAMPLES:
    df_sample = df_merged.sample(n=NUM_SAMPLES, random_state=RANDOM_STATE)
else:
    df_sample = df_merged
print(f"Randomly sampled {len(df_sample)} examples for evaluation.")

survey_data = []
survey_data_with_ids = []

for index, row in df_sample.iterrows():
    explanations_with_source = [
        (row['prediction_ours'], 'Primary-Model'),
        (row['prediction_teacher'], 'FLAN-T5-XXL'),
        (row['prediction_bart'], 'BART-Base')
    ]
    
    random.shuffle(explanations_with_source)
    
    survey_data.append({
        'sample_id': index,
        'history': row['history'],
        'item': row['item'],
        'explanation_A': explanations_with_source[0][0],
        'explanation_B': explanations_with_source[1][0],
        'explanation_C': explanations_with_source[2][0]
    })
    
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

survey_df.to_csv(FINAL_SURVEY_FILE, index=False)
survey_df_with_ids.to_csv(FINAL_SURVEY_FILE_WITH_IDS, index=False)
print(f"\nFinal survey file generated: {FINAL_SURVEY_FILE}")
print(f"Answer key file for analysis generated: {FINAL_SURVEY_FILE_WITH_IDS}")
print("Use the contents of `human_evaluation_survey_data.csv` in your survey tool.")
print("Important: keep `...WITH_IDS.csv` private and do not share it with evaluators.")
