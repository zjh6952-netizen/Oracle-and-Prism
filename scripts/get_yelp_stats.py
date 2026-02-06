import pandas as pd
import os
import json
from tqdm import tqdm

PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Yelp JSON")

BUSINESS_FILE = os.path.join(RAW_DATA_DIR, "yelp_academic_dataset_business.json")
REVIEW_FILE = os.path.join(RAW_DATA_DIR, "yelp_academic_dataset_review.json")

def get_stats():
    print("--- Collecting key Yelp dataset statistics ---")

    try:
        print(f"Reading business file: {BUSINESS_FILE}")
        business_df = pd.read_json(BUSINESS_FILE, lines=True)
        num_items = business_df['business_id'].nunique()
        print(f"Unique businesses (items): {num_items}")
    except FileNotFoundError:
        print(f"!!! Error: business file not found: {BUSINESS_FILE}")
        num_items = "Error"
    except Exception as e:
        print(f"!!! Error while reading business file: {e}")
        num_items = "Error"

    try:
        print(f"\nReading review file: {REVIEW_FILE} (this may take a while)")
        chunk_iter = pd.read_json(REVIEW_FILE, lines=True, chunksize=100000)
        
        unique_users = set()
        for chunk in tqdm(chunk_iter, desc="Scanning user IDs"):
            unique_users.update(chunk['user_id'].unique())
            
        num_users = len(unique_users)
        print(f"Unique users: {num_users}")
    except FileNotFoundError:
        print(f"!!! Error: review file not found: {REVIEW_FILE}")
        num_users = "Error"
    except Exception as e:
        print(f"!!! Error while reading review file: {e}")
        num_users = "Error"

    print("\n\n--- Final Statistics ---")
    print("Use these values in your LaTeX table:")
    print(f"  - # Users (Yelp): {num_users}")
    print(f"  - # Items (Yelp): {num_items}")

if __name__ == "__main__":
    get_stats()
