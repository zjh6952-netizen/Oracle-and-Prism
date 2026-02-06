import pandas as pd
from sklearn.model_selection import train_test_split
import os

PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "yelp_sequences.csv")
TRAIN_FILE_OUTPUT = os.path.join(PROJECT_ROOT, "data", "raw", "yelp_sequences_train.csv")
TEST_FILE_OUTPUT = os.path.join(PROJECT_ROOT, "data", "raw", "yelp_sequences_test.csv")
TEST_SIZE = 0.1
RANDOM_STATE = 42

print(f"Reading raw sequence data: {INPUT_FILE}")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"!!! Error: raw file not found. Run preprocess_movielens.py first.")
    exit()

print(f"Raw dataset has {len(df)} sequences.")
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"After split, train set has {len(train_df)} rows.")
print(f"After split, test set has {len(test_df)} rows.")

train_df.to_csv(TRAIN_FILE_OUTPUT, index=False)
print(f"Train set saved to: {TRAIN_FILE_OUTPUT}")
test_df.to_csv(TEST_FILE_OUTPUT, index=False)
print(f"Test set saved to: {TEST_FILE_OUTPUT}")
print("\nDataset split complete.")
