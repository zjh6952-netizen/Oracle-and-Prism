import pandas as pd
from tqdm import tqdm
import os

PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ml-1m")
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "movielens_sequences.csv")
MIN_SEQUENCE_LENGTH = 10 
MAX_HISTORY_LENGTH = 50 # 核心修复：限制最大历史长度

def prepare_movielens_sequences():
    print("--- 正在处理MovieLens-1M数据集 (已加入历史长度限制) ---")
    try:
        movies_df = pd.read_csv(f"{RAW_DATA_DIR}/movies.dat", sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
        movie_id_to_title = dict(zip(movies_df['MovieID'], movies_df['Title']))
        ratings_df = pd.read_csv(f"{RAW_DATA_DIR}/ratings.dat", sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    except FileNotFoundError as e:
        print(f"!!! 错误: 原始数据文件未找到! {e}")
        return
    
    ratings_df_sorted = ratings_df.sort_values(by=['UserID', 'Timestamp'], ascending=True)
    user_sequences = ratings_df_sorted.groupby('UserID')
    
    final_data = []
    for user_id, user_group in tqdm(user_sequences, desc="处理用户序列"):
        movie_ids = user_group['MovieID'].tolist()
        if len(movie_ids) < MIN_SEQUENCE_LENGTH: continue

        for i in range(1, len(movie_ids)):
            full_history_ids = movie_ids[:i]
            truncated_history_ids = full_history_ids[-MAX_HISTORY_LENGTH:] # 只截取最后50条
            target_id = movie_ids[i]
            
            history_titles = [movie_id_to_title.get(mid, "Unknown") for mid in truncated_history_ids]
            target_title = movie_id_to_title.get(target_id, "Unknown")

            final_data.append({
                "user_id": user_id,
                "history": ", ".join(history_titles),
                "target": target_title
            })
    
    pd.DataFrame(final_data).to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\n🎉 movielens_sequences.csv 文件创建成功！")

if __name__ == "__main__":
    prepare_movielens_sequences()