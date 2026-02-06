import logging
import hashlib
import re
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

USER_VOCAB_SIZE = 331845
ITEM_VOCAB_SIZE = 103912


def _stable_bucket(raw_value, vocab_size):
    """Map arbitrary IDs/text to a stable embedding bucket."""
    if raw_value is None:
        return 0
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return 0

    text = str(raw_value).strip()
    if text == "":
        return 0

    if text.isdigit():
        return int(text) % vocab_size

    # Prefer preserving explicit numeric IDs like item_123 / user_456.
    match = re.search(r"(\d+)$", text)
    if match:
        return int(match.group(1)) % vocab_size

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % vocab_size


def _tokens(text):
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())


def _pseudo_label_quality(source_text, target_text):
    tgt_tokens = _tokens(target_text)
    if not tgt_tokens:
        return 0.0, 0.0, 1.0, 0
    src_tokens = set(_tokens(source_text))
    overlap = sum(1 for tok in tgt_tokens if tok in src_tokens) / len(tgt_tokens)
    unique_ratio = len(set(tgt_tokens)) / len(tgt_tokens)
    repeat_ratio = 1.0 - unique_ratio
    return overlap * 1.8 + unique_ratio * 0.8 - repeat_ratio * 1.2, overlap, repeat_ratio, len(tgt_tokens)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_length, path, max_output_length=None, **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_output_length = max_output_length
        self.path = path
        self.history_max_chars = kwargs.get("history_max_chars")
        self.user_vocab_size = int(kwargs.get("user_vocab_size", USER_VOCAB_SIZE))
        self.item_vocab_size = int(kwargs.get("item_vocab_size", ITEM_VOCAB_SIZE))
        self.filter_pseudo_labels = bool(kwargs.get("filter_pseudo_labels", True))
        self.min_target_tokens = int(kwargs.get("min_target_tokens", 6))
        self.max_target_tokens = int(kwargs.get("max_target_tokens", 80))
        self.max_target_repeat_ratio = float(kwargs.get("max_target_repeat_ratio", 0.55))
        self.min_source_overlap = float(kwargs.get("min_source_overlap", 0.08))
        self.min_quality_score = float(kwargs.get("min_quality_score", 0.20))
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
        except FileNotFoundError:
            logger.error(f"!!! 致命错误: 数据文件未找到，路径: {self.path}")
            self.instances = []
            return

        if "explanation" not in df.columns:
            logger.error("!!! 致命错误: 输入数据缺少 explanation 列。")
            self.instances = []
            return

        df = df.dropna(subset=["explanation"]).copy()
        df["explanation"] = df["explanation"].astype(str)
        df = df[df["explanation"].str.strip() != ""]
        df = df[~df["explanation"].str.contains("Error:", na=False)]

        item_col = None
        for candidate in ["recommended_item", "target", "item"]:
            if candidate in df.columns:
                item_col = candidate
                break

        self.instances = []
        filtered_count = 0
        for row in df.itertuples(index=False):
            history = getattr(row, "history", "")
            history = "" if pd.isna(history) else str(history)
            if self.history_max_chars and len(history) > int(self.history_max_chars):
                history = history[-int(self.history_max_chars):]

            item = ""
            if item_col is not None:
                raw_item = getattr(row, item_col, "")
                item = "" if pd.isna(raw_item) else str(raw_item)
            explanation = str(getattr(row, "explanation", ""))
            user_id = _stable_bucket(getattr(row, "user_id", 1), self.user_vocab_size)
            item_id = _stable_bucket(item, self.item_vocab_size)

            # 【修复】简化输入格式,移除Instruction前缀,避免模型学会复制格式
            # 使用更简洁的提示模板,让模型专注于生成解释
            source_text = f"User History: {history}\nRecommended Item: {item}\nExplanation:"
            if self.filter_pseudo_labels:
                score, overlap, repeat_ratio, token_len = _pseudo_label_quality(source_text, explanation)
                if token_len < self.min_target_tokens or token_len > self.max_target_tokens:
                    filtered_count += 1
                    continue
                if repeat_ratio > self.max_target_repeat_ratio:
                    filtered_count += 1
                    continue
                if overlap < self.min_source_overlap:
                    filtered_count += 1
                    continue
                if score < self.min_quality_score:
                    filtered_count += 1
                    continue

            self.instances.append({
                "source_text": source_text,
                "target_text": explanation,
                "user_id": user_id,
                "item_id": item_id,
            })
        logger.info(f"成功加载并格式化了 {len(self.instances)} 条有效的解释样本。")
        if self.filter_pseudo_labels:
            logger.info(f"伪标签质量过滤掉 {filtered_count} 条样本。")

    def collate_fn(self, batch):
        source_texts = [x["source_text"] for x in batch]
        target_texts = [x["target_text"] for x in batch]
        user_ids = [x["user_id"] for x in batch]
        item_ids = [x["item_id"] for x in batch]

        source_tokenized = self.tokenizer(
            source_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        target_tokenized = self.tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=self.max_output_length,
            return_tensors="pt",
        )

        label_ids = target_tokenized["input_ids"]
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        batch_size, seq_len = source_tokenized["input_ids"].shape

        user_tensor = torch.tensor(user_ids, dtype=torch.long).unsqueeze(1)
        item_tensor = torch.tensor(item_ids, dtype=torch.long).unsqueeze(1)
        enc_user_whole = user_tensor.expand(batch_size, seq_len).contiguous()
        enc_item_whole = item_tensor.expand(batch_size, seq_len).contiguous()
        enc_attrs = torch.zeros(batch_size, seq_len, 1, dtype=torch.long)

        # ===================================================================
        # --- 【核心修复】将target_text包装成二维列表，以匹配train.py评估循环的期望 ---
        # ===================================================================
        formatted_target_texts = [[txt] for txt in target_texts]

        return GenBatch(
            input_text=source_texts,
            target_text=formatted_target_texts, # <-- 使用包装后的二维列表！
            enc_idxs=source_tokenized["input_ids"],
            enc_attn=source_tokenized["attention_mask"],
            lbl_idxs=label_ids,
            enc_user_whole=enc_user_whole,
            enc_item_whole=enc_item_whole,
            enc_attrs=enc_attrs
        )
    
    # 保留空函数定义
    def load_pretrain_data(self): pass
    def load_data(self): pass
    def pretrain_collate_fn(self, batch): pass
