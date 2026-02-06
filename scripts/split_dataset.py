# scripts/split_dataset.py (最终版)
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- 配置 ---
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "yelp_sequences.csv")
TRAIN_FILE_OUTPUT = os.path.join(PROJECT_ROOT, "data", "raw", "yelp_sequences_train.csv")
TEST_FILE_OUTPUT = os.path.join(PROJECT_ROOT, "data", "raw", "yelp_sequences_test.csv")
TEST_SIZE = 0.1
RANDOM_STATE = 42

# --- 主逻辑 ---
print(f"正在读取原始序列数据: {INPUT_FILE}")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"!!! 错误: 原始文件未找到! 请先运行 preprocess_movielens.py")
    exit()

print(f"原始数据集共有 {len(df)} 条序列。")
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"分割后，训练集有 {len(train_df)} 条数据。")
print(f"分割后，测试集有 {len(test_df)} 条数据。")

train_df.to_csv(TRAIN_FILE_OUTPUT, index=False)
print(f"训练集已保存到: {TRAIN_FILE_OUTPUT}")
test_df.to_csv(TEST_FILE_OUTPUT, index=False)
print(f"测试集已保存到: {TEST_FILE_OUTPUT}")
print("\n🎉 数据集分割完成！")