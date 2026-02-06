# scripts/get_yelp_stats.py (最终兼容版)
import pandas as pd
import os
import json
from tqdm import tqdm

# --- 配置 ---
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Yelp JSON")

BUSINESS_FILE = os.path.join(RAW_DATA_DIR, "yelp_academic_dataset_business.json")
REVIEW_FILE = os.path.join(RAW_DATA_DIR, "yelp_academic_dataset_review.json")

def get_stats():
    print("--- 正在统计 Yelp 数据集核心参数 ---")

    # --- 1. 统计独立商家 (Items) 数量 ---
    try:
        print(f"正在读取商家文件: {BUSINESS_FILE}")
        # 【核心修复】移除 usecols 参数，先把整个文件读进来
        business_df = pd.read_json(BUSINESS_FILE, lines=True)
        # 然后再只对 'business_id' 这一列进行操作
        num_items = business_df['business_id'].nunique()
        print(f"独立商家 (Items) 数量: {num_items}")
    except FileNotFoundError:
        print(f"!!! 错误: 找不到商家文件! {BUSINESS_FILE}")
        num_items = "Error"
    except Exception as e:
        print(f"!!! 读取商家文件时出错: {e}")
        num_items = "Error"

    # --- 2. 统计独立用户 (Users) 数量 ---
    try:
        print(f"\n正在读取评论文件: {REVIEW_FILE} (这可能需要一些时间)")
        # 【核心修复】同样，在分块读取时，也移除 usecols 参数
        chunk_iter = pd.read_json(REVIEW_FILE, lines=True, chunksize=100000)
        
        unique_users = set()
        # 我们依然可以高效地扫描
        for chunk in tqdm(chunk_iter, desc="扫描用户ID"):
            unique_users.update(chunk['user_id'].unique())
            
        num_users = len(unique_users)
        print(f"独立用户 (Users) 数量: {num_users}")
    except FileNotFoundError:
        print(f"!!! 错误: 找不到评论文件! {REVIEW_FILE}")
        num_users = "Error"
    except Exception as e:
        print(f"!!! 读取评论文件时出错: {e}")
        num_users = "Error"

    print("\n\n--- 最终统计结果 ---")
    print("请将以下数字填入你的LaTeX表格中:")
    print(f"  - # Users (Yelp): {num_users}")
    print(f"  - # Items (Yelp): {num_items}")

if __name__ == "__main__":
    get_stats()