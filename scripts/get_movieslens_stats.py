import pandas as pd
import os

PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ml-1m")
RATINGS_FILE = os.path.join(RAW_DATA_DIR, "ratings.dat")
MOVIES_FILE = os.path.join(RAW_DATA_DIR, "movies.dat")

def get_stats():
    print("--- Auditing key MovieLens-1M dataset statistics ---")
    
    num_users = "N/A"
    num_items = "N/A"
    num_train_seq = "N/A"
    num_test_seq = "N/A"

    try:
        print(f"Counting users from {RATINGS_FILE}...")
        ratings_df = pd.read_csv(
            RATINGS_FILE, sep='::', engine='python',
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        num_users = ratings_df['UserID'].nunique()
        
        print(f"Counting items from {MOVIES_FILE}...")
        movies_df = pd.read_csv(
            MOVIES_FILE, sep='::', engine='python',
            names=['MovieID', 'Title', 'Genres'], encoding='latin-1'
        )
        num_items = movies_df['MovieID'].nunique()

    except Exception as e:
        print(f"!!! Error while reading raw .dat files: {e}")

    print("\n\n--- Final Audit Results ---")
    print("Please verify these numbers against your paper table:")
    print(f"  - # Users (MovieLens-1M): {num_users}")
    print(f"  - # Items (MovieLens-1M): {num_items}")
    
if __name__ == "__main__":
    get_stats()
