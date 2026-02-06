import pandas as pd
from tqdm import tqdm
import os
import json

# --- 配置 ---
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Yelp JSON")
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "yelp_sequences.csv")

MIN_INTERACTIONS_PER_USER = 15 # Yelp数据量大，我们可以提高门槛
MAX_HISTORY_LENGTH = 50 

def prepare_yelp_sequences():
    print("--- 正在处理 Yelp 数据集 ---")
    
    # --- 1. 加载商家元数据，创建ID到名称的映射 ---
    business_path = os.path.join(RAW_DATA_DIR, "yelp_academic_dataset_business.json")
    print(f"加载商家数据: {business_path}")
    business_data = []
    with open(business_path, 'r', encoding='utf-8') as f:
        for line in f:
            business_data.append(json.loads(line))
    business_df = pd.DataFrame(business_data)
    # 我们创建一个ID -> (名称, 类别)的映射
    business_df['categories'] = business_df['categories'].fillna('Unknown')
    id_to_meta = dict(zip(business_df['business_id'], zip(business_df['name'], business_df['categories'])))
    print(f"成功加载 {len(id_to_meta)} 个商家的信息。")
        
    # --- 2. 加载评论数据 (核心) ---
    review_path = os.path.join(RAW_DATA_DIR, "yelp_academic_dataset_review.json")
    print(f"加载评论数据: {review_path} (这可能需要一些时间)")
    review_df = pd.read_json(review_path, lines=True)
    # 为了加速，我们可以只取一部分数据，比如评分最高的
    review_df = review_df[review_df['stars'] >= 4.0]
    # 为了演示，我们先取一个子集
    # user_counts = review_df['user_id'].value_counts()
    # active_users = user_counts[user_counts >= MIN_INTERACTIONS_PER_USER].index
    # review_df = review_df[review_df['user_id'].isin(active_users)]
    
    # --- 3. 排序与分组 ---
    print("正在按用户和日期对评论进行排序...")
    review_df['date'] = pd.to_datetime(review_df['date'])
    review_df_sorted = review_df.sort_values(by=['user_id', 'date'], ascending=True)
    user_sequences = review_df_sorted.groupby('user_id')

    # --- 4. 生成序列 ---
    print("正在生成 (历史, 目标) 序列...")
    final_data = []
    for user_id, user_group in tqdm(user_sequences, desc="处理用户序列"):
        business_ids = user_group['business_id'].tolist()
        if len(business_ids) < MIN_INTERACTIONS_PER_USER:
            continue

        for i in range(1, len(business_ids)):
            full_history_ids = business_ids[:i]
            truncated_history_ids = full_history_ids[-MAX_HISTORY_LENGTH:]
            target_id = business_ids[i]
            
            # 我们不仅加入名称，也加入类别，让历史更丰富
            def get_name_cat(biz_id):
                name, cat = id_to_meta.get(biz_id, ("Unknown", "Unknown"))
                return f"{name} ({cat.split(',')[0]})" # 只取第一个类别

            history_str = ", ".join([get_name_cat(bid) for bid in truncated_history_ids])
            target_str = get_name_cat(target_id)

            final_data.append({
                "user_id": user_id,
                "history": history_str,
                "target": target_str
            })

    # --- 5. 保存 ---
    final_df = pd.DataFrame(final_data)
    print(f"\n处理完成，共生成 {len(final_df)} 条有效序列数据。")
    final_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"🎉 yelp_sequences.csv 文件创建成功！")

if __name__ == "__main__":
    prepare_yelp_sequences()