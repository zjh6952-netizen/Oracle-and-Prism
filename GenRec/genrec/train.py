import os, sys, json, logging, time, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from model import GenerativeModel
from data import Dataset
from argparse import ArgumentParser, Namespace
import datetime
import math
from utils import *

# ！！注意！！：文件顶部没有任何关于TOKENIZERS_PARALLELISM的设置

def get_explanation_score(epoch, references, predictions):
    # 【核心修复】只有在需要时，才在函数内部导入
    try:
        import evaluate
        rouge_scorer = evaluate.load("/root/autodl-tmp/GenRec_Explainer_Project/offline_metrics/rouge")
    except Exception as e:
        print(f"加载ROUGE评估器失败: {e}")
        return 0.0, "{'Error': 'Scorer not available'}"
    flat_references = [item for sublist in references for item in sublist]
    if len(predictions) != len(flat_references):
        print(f"警告: 预测({len(predictions)})与参考({len(flat_references)})数量不匹配。")
        min_len = min(len(predictions), len(flat_references))
        predictions = predictions[:min_len]
        flat_references = flat_references[:min_len]
    try:
        result = rouge_scorer.compute(predictions=predictions, references=flat_references)
        rouge_l_score = result.get('rougeL', 0.0)
    except Exception as e:
        print(f"计算ROUGE分数时出错: {e}")
        rouge_l_score = 0.0
    post_fix = { "Epoch": epoch, "ROUGE-L": f'{rouge_l_score:.4f}' }
    print(post_fix)
    return rouge_l_score, str(post_fix)

# --- GenRec原始代码结构 ---
parser = ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-pmp', '--pretrained_model_path', type=str, required=False)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config.update(args.__dict__)
config = Namespace(**config)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir): os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]',
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
summarizer = Summarizer(output_dir)
torch.cuda.set_device(config.gpu_device)
with open(os.path.join(output_dir, 'config.json'), 'w') as fp: json.dump(vars(config), fp, indent=4)
best_model_path = os.path.join(output_dir, 'best_model.mdl')
dev_prediction_path = os.path.join(output_dir, 'pred.dev.txt')
tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True, add_prefix_space=True)
special_tokens = ['<mask>']
tokenizer.add_tokens(special_tokens)
train_set = Dataset(tokenizer, config.max_length, config.train_file, config.max_output_length)
dev_set = Dataset(tokenizer, config.max_length, config.dev_file, config.max_output_length)
train_batch_num = len(train_set) // (config.train_batch_size * config.accumulate_step) + \
                  (len(train_set) % (config.train_batch_size * config.accumulate_step) != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
model = GenerativeModel(config, tokenizer)
model_path = args.pretrained_model_path
if model_path is not None:
    logger.info("Loading model from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=f'cuda:{config.gpu_device}'), strict=False)
    logger.info("Loaded model from {}".format(model_path))
model.cuda(device=config.gpu_device)
param_groups = [{'params': model.parameters(), 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=train_batch_num * config.warmup_epoch,
                                           num_training_steps=train_batch_num * config.max_epoch)
logger.info("Start training ...")
summarizer_step = 0
best_dev_score = 0.0
collate_fn = train_set.collate_fn
for epoch in range(1, config.max_epoch + 1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")
    progress = tqdm.tqdm(total=train_batch_num, ncols=75, desc='Train {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    # ！！注意！！：我们保留了原始的并行化数据加载
    for batch_idx, batch in enumerate(
            DataLoader(train_set, batch_size=config.train_batch_size,
                       shuffle=True, drop_last=False, collate_fn=collate_fn)):
        try:
            gpu_batch = move_to_cuda(batch)
            loss = model(gpu_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("OOM in forward/backward pass")
                optimizer.zero_grad(); torch.cuda.empty_cache()
                continue
            else: raise e
        if batch_idx % 1000 == 0:
            logger.info('epoch {} batch_idx {} loss {}'.format(epoch, batch_idx, loss.item()))
        summarizer.scalar_summary('train/loss', loss, summarizer_step)
        summarizer_step += 1
        loss = loss * (1 / config.accumulate_step)
        loss.backward()
        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    progress.close()
    logger.info('Saving epoch_{} model...'.format(epoch))
    torch.save(model.state_dict(), os.path.join(output_dir, f'epoch_{epoch}.mdl'))
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev {}'.format(epoch))
    model.eval()
    torch.cuda.empty_cache()
    all_predictions = []
    all_references = []
    for batch_idx, batch in enumerate(DataLoader(dev_set, batch_size=config.eval_batch_size,
                                                 shuffle=False, collate_fn=dev_set.collate_fn)):
        progress.update(1)
        try:
            gpu_batch = move_to_cuda(batch)
            predictions = model.predict(gpu_batch, max_length=config.max_output_length)
            all_predictions.extend(predictions)
            all_references.extend(batch.target_text)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("OOM in evaluation")
                torch.cuda.empty_cache()
                continue
            else: raise e
    progress.close()
    current_score, post_fix = get_explanation_score(epoch, all_references, all_predictions)
    if current_score > best_dev_score:
        best_dev_score = current_score
        logger.info(f"🎉 New best dev score (ROUGE-L) found: {best_dev_score:.4f} at epoch {epoch}")
        logger.info('Saving best dev model to {}'.format(best_model_path))
        torch.save(model.state_dict(), best_model_path)
        logger.info('Writing best dev prediction to {}'.format(dev_prediction_path))
        with open(dev_prediction_path, 'w') as fw:
            fw.writelines(post_fix + '\n')
            for ref_list, pred in zip(all_references, all_predictions):
                ref = ref_list[0] if ref_list else ""
                fw.writelines(f"GOLD: {ref}\nPRED: {pred}\n\n")
logger.info(log_path)
logger.info("Done!")