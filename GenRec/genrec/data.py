import logging
import torch
from collections import namedtuple
import pandas as pd

# 获取一个日志记录器
logger = logging.getLogger(__name__)

# 定义GenBatch结构
gen_batch_fields = [
    'input_text', 'target_text', 'enc_idxs', 'enc_attn', 
    'dec_idxs', 'dec_attn', 'lbl_idxs', 'raw_lbl_idxs', 
    'enc_user_whole', 'enc_item_whole', 'enc_attrs'
]
GenBatch = namedtuple('GenBatch', field_names=gen_batch_fields, defaults=[None] * len(gen_batch_fields))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_length, path, max_output_length=None, **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_output_length = max_output_length
        self.path = path
        self.load_explanation_data()

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

    def load_explanation_data(self):
        logger.info(f"为解释任务加载数据 (BART兼容模式): {self.path}")
        try:
            logger.info("--> 正在使用 Pandas 读取大型CSV文件，请耐心等待...")
            df = pd.read_csv(self.path)
            logger.info(f"--> CSV文件读取完毕！共 {len(df)} 行。现在开始格式化...")
            
            df.dropna(subset=['explanation'], inplace=True)
            df = df[~df['explanation'].str.contains("Error:")]
        except FileNotFoundError:
            logger.error(f"!!! 致命错误: 数据文件未找到，路径: {self.path}")
            self.instances = []
            return

        self.instances = []
        for index, row in df.iterrows():
            history = str(row.get('history', ''))
            # 【优化】限制历史记录长度,避免输入过长
            # 只保留最近的10个项目,提高模型专注度
            history_items = [item.strip() for item in history.split(',')]
            if len(history_items) > 10:
                history_items = history_items[-10:]  # 保留最近10个
            history = ', '.join(history_items)
            
            item = str(row.get('recommended_item', ''))
            explanation = str(row.get('explanation', ''))
            user_id = int(row.get('user_id', 1))

            # 【修复】简化输入格式,移除Instruction前缀,避免模型学会复制格式
            # 使用更简洁的提示模板,让模型专注于生成解释
            source_text = f"User History: {history}\nRecommended Item: {item}\nExplanation:"
            
            self.instances.append({ 
                "source_text": source_text, 
                "target_text": explanation,
                "user_id": user_id 
            })
        logger.info(f"成功加载并格式化了 {len(self.instances)} 条有效的解释样本。")

    def collate_fn(self, batch):
        source_texts = [x['source_text'] for x in batch]
        target_texts = [x['target_text'] for x in batch]
        user_ids = [x['user_id'] for x in batch]
        
        source_tokenized = self.tokenizer(source_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        target_tokenized = self.tokenizer(target_texts, padding='max_length', truncation=True, max_length=self.max_output_length, return_tensors='pt')
        
        label_ids = target_tokenized['input_ids']
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        
        batch_size, seq_len = source_tokenized['input_ids'].shape
        
        enc_user_whole = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for i in range(batch_size):
            enc_user_whole[i, :] = user_ids[i]

        enc_item_whole = torch.zeros(batch_size, seq_len, dtype=torch.long)
        enc_attrs = torch.zeros(batch_size, seq_len, 1, dtype=torch.long)

        # ===================================================================
        # --- 【核心修复】将target_text包装成二维列表，以匹配train.py评估循环的期望 ---
        # ===================================================================
        formatted_target_texts = [[txt] for txt in target_texts]

        return GenBatch(
            input_text=source_texts,
            target_text=formatted_target_texts, # <-- 使用包装后的二维列表！
            enc_idxs=source_tokenized['input_ids'],
            enc_attn=source_tokenized['attention_mask'],
            lbl_idxs=label_ids,
            enc_user_whole=enc_user_whole,
            enc_item_whole=enc_item_whole,
            enc_attrs=enc_attrs
        )
    
    # 保留空函数定义
    def load_pretrain_data(self): pass
    def load_data(self): pass
    def pretrain_collate_fn(self, batch): pass