import json
import logging
import math
import os
import pprint
import random
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from data import Dataset
from model import GenerativeModel
from utils import Summarizer, move_to_cuda


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LEGACY_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"


def cfg(config, key, default):
    return getattr(config, key, default)


def resolve_path(path_value, base_dir=PROJECT_ROOT):
    if path_value is None:
        return None

    path_value = str(path_value)
    if os.path.exists(path_value):
        return path_value

    if os.path.isabs(path_value):
        if path_value.startswith(LEGACY_ROOT):
            migrated = path_value.replace(LEGACY_ROOT, base_dir, 1)
            if os.path.exists(migrated) or not os.path.exists(path_value):
                return migrated
        return path_value

    cwd_candidate = os.path.abspath(path_value)
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    project_candidate = os.path.abspath(os.path.join(base_dir, path_value))
    if os.path.exists(project_candidate):
        return project_candidate

    return project_candidate


def build_rouge_scorer(config, logger):
    try:
        import evaluate
    except Exception as err:
        logger.warning("evaluate 导入失败，将跳过ROUGE评估: %s", err)
        return None

    rouge_path = resolve_path(
        cfg(config, "rouge_metric_path", os.path.join(PROJECT_ROOT, "offline_metrics", "rouge"))
    )
    try:
        return evaluate.load(rouge_path)
    except Exception as err:
        logger.warning("本地ROUGE加载失败(%s)，尝试默认rouge: %s", rouge_path, err)

    try:
        return evaluate.load("rouge")
    except Exception as err:
        logger.warning("默认rouge加载失败: %s", err)
        return None


def get_explanation_score(epoch, references, predictions, rouge_scorer, logger):
    if rouge_scorer is None:
        post_fix = {"Epoch": epoch, "ROUGE-L": "0.0000"}
        return 0.0, str(post_fix)

    flat_references = [item for sublist in references for item in sublist]
    if len(predictions) != len(flat_references):
        logger.warning(
            "预测(%s)与参考(%s)数量不匹配，将按较短长度截断。",
            len(predictions),
            len(flat_references),
        )
        min_len = min(len(predictions), len(flat_references))
        predictions = predictions[:min_len]
        flat_references = flat_references[:min_len]

    try:
        result = rouge_scorer.compute(predictions=predictions, references=flat_references)
        rouge_l_score = float(result.get("rougeL", 0.0))
    except Exception as err:
        logger.warning("计算ROUGE分数失败: %s", err)
        rouge_l_score = 0.0

    post_fix = {"Epoch": epoch, "ROUGE-L": f"{rouge_l_score:.4f}"}
    logger.info(post_fix)
    return rouge_l_score, str(post_fix)


def make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, collate_fn, persistent_workers, prefetch_factor):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-pmp", "--pretrained_model_path", type=str, required=False)
    args = parser.parse_args()

    with open(args.config) as fp:
        config = json.load(fp)
    config.update(args.__dict__)
    config = Namespace(**config)

    # Seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = bool(cfg(config, "cudnn_benchmark", True))

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(config.gpu_device)
        device = torch.device(f"cuda:{config.gpu_device}")
    else:
        device = torch.device("cpu")

    # Resolve paths (compatible with both legacy absolute paths and local repo paths)
    config.train_file = resolve_path(config.train_file)
    config.dev_file = resolve_path(config.dev_file)
    config.model_name = resolve_path(config.model_name)
    config.output_dir = resolve_path(config.output_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = os.path.join(config.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info("\n%s", pprint.pformat(vars(config), indent=4))
    logger.info("Training device: %s", device)

    summarizer = Summarizer(output_dir)
    with open(os.path.join(output_dir, "config.json"), "w") as fp:
        json.dump(vars(config), fp, indent=4)

    best_model_path = os.path.join(output_dir, "best_model.mdl")
    dev_prediction_path = os.path.join(output_dir, "pred.dev.txt")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=bool(cfg(config, "local_files_only", True)),
        add_prefix_space=True,
    )
    tokenizer.add_tokens(["<mask>"])

    dataset_kwargs = {
        "history_max_chars": cfg(config, "history_max_chars", None),
        "user_vocab_size": cfg(config, "user_vocab_size", 331845),
        "item_vocab_size": cfg(config, "item_vocab_size", 103912),
        "filter_pseudo_labels": cfg(config, "filter_pseudo_labels", True),
        "min_target_tokens": cfg(config, "min_target_tokens", 6),
        "max_target_tokens": cfg(config, "max_target_tokens", 80),
        "max_target_repeat_ratio": cfg(config, "max_target_repeat_ratio", 0.55),
        "min_source_overlap": cfg(config, "min_source_overlap", 0.08),
        "min_quality_score": cfg(config, "min_quality_score", 0.20),
    }
    train_set = Dataset(tokenizer, config.max_length, config.train_file, config.max_output_length, **dataset_kwargs)
    dev_set = Dataset(tokenizer, config.max_length, config.dev_file, config.max_output_length, **dataset_kwargs)
    if len(train_set) == 0 or len(dev_set) == 0:
        raise ValueError("训练集或验证集为空，请检查数据路径和CSV格式。")

    train_num_workers = int(cfg(config, "train_num_workers", 4 if use_cuda else 0))
    eval_num_workers = int(cfg(config, "eval_num_workers", 2 if use_cuda else 0))
    pin_memory = bool(cfg(config, "pin_memory", use_cuda))
    persistent_workers = bool(cfg(config, "persistent_workers", True))
    prefetch_factor = int(cfg(config, "prefetch_factor", 2))

    train_loader = make_dataloader(
        dataset=train_set,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        collate_fn=train_set.collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    dev_loader = make_dataloader(
        dataset=dev_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=pin_memory,
        collate_fn=dev_set.collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    accumulate_step = max(1, int(config.accumulate_step))
    train_batch_num = math.ceil(len(train_loader) / accumulate_step)
    dev_batch_num = len(dev_loader)

    model = GenerativeModel(config, tokenizer)
    model_path = resolve_path(args.pretrained_model_path) if args.pretrained_model_path else None
    if model_path is not None:
        logger.info("Loading model from %s", model_path)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        logger.info("Loaded model from %s", model_path)
    model.to(device)

    param_groups = [{"params": model.parameters(), "lr": config.learning_rate, "weight_decay": config.weight_decay}]
    optimizer = AdamW(params=param_groups)
    schedule = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_batch_num * config.warmup_epoch),
        num_training_steps=int(train_batch_num * config.max_epoch),
    )

    amp_mode = str(cfg(config, "amp_mode", "none")).lower()
    if use_cuda and amp_mode == "none" and torch.cuda.is_bf16_supported():
        amp_mode = "bf16"
    if amp_mode == "bf16" and (not use_cuda or not torch.cuda.is_bf16_supported()):
        amp_mode = "fp16" if use_cuda else "none"
    use_amp = use_cuda and amp_mode in {"bf16", "fp16"}
    use_fp16 = use_amp and amp_mode == "fp16"
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    logger.info("AMP mode: %s", amp_mode)

    rouge_scorer = build_rouge_scorer(config, logger)

    logger.info("Start training ...")
    summarizer_step = 0
    best_dev_score = -1.0
    save_each_epoch = bool(cfg(config, "save_each_epoch", True))
    log_every = int(cfg(config, "log_every", 100))
    eval_num_beams = int(cfg(config, "eval_num_beams", 5))
    eval_repetition_penalty = float(cfg(config, "eval_repetition_penalty", 1.2))
    eval_no_repeat_ngram_size = int(cfg(config, "eval_no_repeat_ngram_size", 3))
    eval_length_penalty = float(cfg(config, "eval_length_penalty", 1.0))
    eval_min_new_tokens = int(cfg(config, "eval_min_new_tokens", 0))

    for epoch in range(1, config.max_epoch + 1):
        logger.info(log_path)
        logger.info("Epoch %s", epoch)
        progress = tqdm.tqdm(total=train_batch_num, ncols=90, desc=f"Train {epoch}")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        grad_accum_counter = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            try:
                gpu_batch = move_to_cuda(batch) if use_cuda else batch
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    loss = model(gpu_batch)
            except RuntimeError as err:
                if "out of memory" in str(err).lower():
                    logger.warning("OOM in train step, batch=%s", batch_idx)
                    optimizer.zero_grad(set_to_none=True)
                    grad_accum_counter = 0
                    if use_cuda:
                        torch.cuda.empty_cache()
                    continue
                raise err

            loss_value = float(loss.detach().item())
            if batch_idx % log_every == 0 or batch_idx == 1:
                logger.info("epoch=%s batch=%s loss=%.6f", epoch, batch_idx, loss_value)
            summarizer.scalar_summary("train/loss", loss_value, summarizer_step)
            summarizer_step += 1

            loss = loss / accumulate_step
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_accum_counter += 1
            should_step = (grad_accum_counter >= accumulate_step) or (batch_idx == len(train_loader))
            if not should_step:
                continue

            if float(config.grad_clipping) > 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clipping))

            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            grad_accum_counter = 0
            progress.update(1)

        progress.close()

        if save_each_epoch:
            epoch_model_path = os.path.join(output_dir, f"epoch_{epoch}.mdl")
            logger.info("Saving epoch model to %s", epoch_model_path)
            torch.save(model.state_dict(), epoch_model_path)

        progress = tqdm.tqdm(total=dev_batch_num, ncols=90, desc=f"Dev {epoch}")
        model.eval()
        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in dev_loader:
                progress.update(1)
                try:
                    gpu_batch = move_to_cuda(batch) if use_cuda else batch
                    predictions = model.predict(
                        gpu_batch,
                        num_beams=eval_num_beams,
                        max_length=config.max_output_length,
                        repetition_penalty=eval_repetition_penalty,
                        no_repeat_ngram_size=eval_no_repeat_ngram_size,
                        length_penalty=eval_length_penalty,
                        min_new_tokens=eval_min_new_tokens,
                    )
                    all_predictions.extend(predictions)
                    all_references.extend(batch.target_text)
                except RuntimeError as err:
                    if "out of memory" in str(err).lower():
                        logger.warning("OOM in evaluation step")
                        if use_cuda:
                            torch.cuda.empty_cache()
                        continue
                    raise err

        progress.close()
        current_score, post_fix = get_explanation_score(epoch, all_references, all_predictions, rouge_scorer, logger)

        if current_score > best_dev_score:
            best_dev_score = current_score
            logger.info("New best dev ROUGE-L=%.4f at epoch %s", best_dev_score, epoch)
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saved best model to %s", best_model_path)
            with open(dev_prediction_path, "w") as fw:
                fw.writelines(post_fix + "\n")
                for ref_list, pred in zip(all_references, all_predictions):
                    ref = ref_list[0] if ref_list else ""
                    fw.writelines(f"GOLD: {ref}\nPRED: {pred}\n\n")

    logger.info(log_path)
    logger.info("Done! Best dev ROUGE-L=%.4f", best_dev_score)
    summarizer.writer.close()


if __name__ == "__main__":
    main()
