import torch
import torch.nn as nn
from transformers import AutoConfig
import re
# 确保从正确的本地路径导入魔改的BART
from genrec.bart.model import BartForConditionalGeneration

class GenerativeModel(nn.Module):
    def __init__(self, config, tokenizer, **kwargs): # 使用**kwargs增加灵活性
        super().__init__()
        self.tokenizer = tokenizer
        self.model_config = AutoConfig.from_pretrained(config.model_name, local_files_only=True)
        self.model = BartForConditionalGeneration.from_pretrained(
            config.model_name,
            config=self.model_config,
            local_files_only=True
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def forward(self, batch):
        """
        此函数保持原样，以确保与train.py的训练循环100%兼容。
        """
        outputs = self.model(
            input_ids=batch.enc_idxs,
            attention_mask=batch.enc_attn,
            enc_user_whole=batch.enc_user_whole,
            enc_item_whole=batch.enc_item_whole,
            enc_attrs=batch.enc_attrs,
            decoder_input_ids=batch.dec_idxs,
            decoder_attention_mask=batch.dec_attn,
            labels=batch.lbl_idxs,
            return_dict=True
        )
        loss = outputs['loss']
        return loss

    def predict(
        self,
        batch,
        num_beams=5,
        max_length=150,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        min_new_tokens=0,
    ):
        """
        【最终修复版】
        为解释任务重写的predict方法。
        我们强制只为每个输入返回1个最佳输出，以确保数量匹配。
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            enc_kwargs = {
                "enc_user_whole": batch.enc_user_whole,
                "enc_item_whole": batch.enc_item_whole,
                "enc_attrs": batch.enc_attrs
            }
            
            outputs = self.model.generate(
                input_ids=batch.enc_idxs,
                attention_mask=batch.enc_attn,
                num_beams=num_beams,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                min_new_tokens=min_new_tokens,
                # --- 【核心修复】我们只要求返回1个序列！ ---
                num_return_sequences=1, 
                return_dict_in_generate=True,
                **enc_kwargs
            )

        # 解码所有生成的序列
        generated_explanations = self.tokenizer.batch_decode(
            outputs['sequences'], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        if was_training:
            self.train()
        return generated_explanations
