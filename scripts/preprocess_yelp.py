import pandas as pd
from tqdm import tqdm
import os
import json

PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Yelp JSON")
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "yelp_sequences.csv")

MIN_INTERACTIONS_PER_USER = 15
MAX_HISTORY_LENGTH = 50 

def prepare_yelp_sequences():
    print("--- Processing Yelp dataset ---")
    
    business_path = os.path.join(RAW_DATA_DIR, "yelp_academic_dataset_business.json")
    print(f"Loading business data: {business_path}")
    business_data = []
    with open(business_path, 'r', encoding='utf-8') as f:
        for line in f:
            business_data.append(json.loads(line))
    business_df = pd.DataFrame(business_data)
    business_df['categories'] = business_df['categories'].fillna('Unknown')
    id_to_meta = dict(zip(business_df['business_id'], zip(business_df['name'], business_df['categories'])))
    print(f"Loaded metadata for {len(id_to_meta)} businesses.")
        
    review_path = os.path.join(RAW_DATA_DIR, "yelp_academic_dataset_review.json")
    print(f"Loading review data: {review_path} (this may take a while)")
    review_df = pd.read_json(review_path, lines=True)
    review_df = review_df[review_df['stars'] >= 4.0]
    # user_counts = review_df['user_id'].value_counts()
    # active_users = user_counts[user_counts >= MIN_INTERACTIONS_PER_USER].index
    # review_df = review_df[review_df['user_id'].isin(active_users)]
    
    print("Sorting reviews by user and date...")
    review_df['date'] = pd.to_datetime(review_df['date'])
    review_df_sorted = review_df.sort_values(by=['user_id', 'date'], ascending=True)
    user_sequences = review_df_sorted.groupby('user_id')

    print("Generating (history, target) sequences...")
    final_data = []
    for user_id, user_group in tqdm(user_sequences, desc="Processing user sequences"):
        business_ids = user_group['business_id'].tolist()
        if len(business_ids) < MIN_INTERACTIONS_PER_USER:
            continue

        for i in range(1, len(business_ids)):
            full_history_ids = business_ids[:i]
            truncated_history_ids = full_history_ids[-MAX_HISTORY_LENGTH:]
            target_id = business_ids[i]
            
            def get_name_cat(biz_id):
                name, cat = id_to_meta.get(biz_id, ("Unknown", "Unknown"))
                return f"{name} ({cat.split(',')[0]})"

            history_str = ", ".join([get_name_cat(bid) for bid in truncated_history_ids])
            target_str = get_name_cat(target_id)

            final_data.append({
                "user_id": user_id,
                "history": history_str,
                "target": target_str
            })

    final_df = pd.DataFrame(final_data)
    print(f"\nDone. Generated {len(final_df)} valid sequence rows.")
    final_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"yelp_sequences.csv created successfully.")

if __name__ == "__main__":
    prepare_yelp_sequences()
