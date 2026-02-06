import pandas as pd
from tqdm import tqdm
import os

PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ml-1m")
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "movielens_sequences.csv")
MIN_SEQUENCE_LENGTH = 10 
MAX_HISTORY_LENGTH = 50

def prepare_movielens_sequences():
    print("--- Processing MovieLens-1M dataset (with history-length limit) ---")
    try:
        movies_df = pd.read_csv(f"{RAW_DATA_DIR}/movies.dat", sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
        movie_id_to_title = dict(zip(movies_df['MovieID'], movies_df['Title']))
        ratings_df = pd.read_csv(f"{RAW_DATA_DIR}/ratings.dat", sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    except FileNotFoundError as e:
        print(f"!!! Error: raw data file not found: {e}")
        return
    
    ratings_df_sorted = ratings_df.sort_values(by=['UserID', 'Timestamp'], ascending=True)
    user_sequences = ratings_df_sorted.groupby('UserID')
    
    final_data = []
    for user_id, user_group in tqdm(user_sequences, desc="Processing user sequences"):
        movie_ids = user_group['MovieID'].tolist()
        if len(movie_ids) < MIN_SEQUENCE_LENGTH: continue

        for i in range(1, len(movie_ids)):
            full_history_ids = movie_ids[:i]
            truncated_history_ids = full_history_ids[-MAX_HISTORY_LENGTH:]
            target_id = movie_ids[i]
            
            history_titles = [movie_id_to_title.get(mid, "Unknown") for mid in truncated_history_ids]
            target_title = movie_id_to_title.get(target_id, "Unknown")

            final_data.append({
                "user_id": user_id,
                "history": ", ".join(history_titles),
                "target": target_title
            })
    
    pd.DataFrame(final_data).to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\nmovielens_sequences.csv created successfully.")

if __name__ == "__main__":
    prepare_movielens_sequences()
