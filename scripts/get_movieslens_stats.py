import pandas as pd
import os

# --- 配置 ---
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"

# 我们也需要原始的评分文件，来精确计算总用户和物品数
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ml-1m")
RATINGS_FILE = os.path.join(RAW_DATA_DIR, "ratings.dat")
MOVIES_FILE = os.path.join(RAW_DATA_DIR, "movies.dat")

def get_stats():
    print("--- 正在审计 MovieLens-1M 数据集的核心参数 ---")
    
    num_users = "N/A"
    num_items = "N/A"
    num_train_seq = "N/A"
    num_test_seq = "N/A"

    # --- 1. 统计总用户和物品数 (从最原始的文件) ---
    try:
        print(f"正在从 {RATINGS_FILE} 统计用户数...")
        ratings_df = pd.read_csv(
            RATINGS_FILE, sep='::', engine='python',
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        num_users = ratings_df['UserID'].nunique()
        
        print(f"正在从 {MOVIES_FILE} 統計物品数...")
        movies_df = pd.read_csv(
            MOVIES_FILE, sep='::', engine='python',
            names=['MovieID', 'Title', 'Genres'], encoding='latin-1'
        )
        num_items = movies_df['MovieID'].nunique()

    except Exception as e:
        print(f"!!! 读取原始.dat文件时出错: {e}")

    print("\n\n--- 最终审计结果 ---")
    print("请将以下数字与你论文表格中的数字进行核对:")
    print(f"  - # Users (MovieLens-1M): {num_users}")
    print(f"  - # Items (MovieLens-1M): {num_items}")
    
if __name__ == "__main__":
    get_stats()