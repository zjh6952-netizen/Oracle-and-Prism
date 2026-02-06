#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use GPT as a judge to compute GPT-score for recommendation explanations.

Mode A (default): generate explanations from trained model weights, then score.
  python scripts/evaluate_gpt_score.py \
    --mode from_model \
    --model_base_path /path/to/bart-base \
    --model_weights_path /path/to/best_model.mdl \
    --test_data_path /path/to/explanation_dataset_test.csv \
    --output_csv results/our_model_with_gpt_score.csv \
    --summary_json results/our_model_gpt_score_summary.json

Mode B: score an existing prediction CSV.
  python scripts/evaluate_gpt_score.py \
    --mode from_csv \
    --input_csv results/evaluation_results_our_model.csv \
    --output_csv results/evaluation_results_our_model_with_gpt_score.csv \
    --summary_json results/evaluation_results_our_model_gpt_score_summary.json
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, Optional, Tuple

import pandas as pd

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: openai. Please run `pip install openai` first."
    ) from exc


SYSTEM_PROMPT = """You are a evaluator for recommendation explanations.
Given user history, recommended item, and model explanation, score the explanation.

Return ONLY JSON with these fields:
{
  "faithfulness": float,      // 1.0 to 5.0
  "personalization": float,   // 1.0 to 5.0
  "persuasiveness": float,    // 1.0 to 5.0
  "fluency": float,           // 1.0 to 5.0
  "overall": float,           // 1.0 to 5.0
  "reason": string            // <= 40 words
}

Rubric:
- faithfulness: does not hallucinate; supported by history/item.
- personalization: tailored to this specific user history.
- persuasiveness: recommendation sounds convincing and useful.
- fluency: grammatical, coherent, concise.
- overall: holistic quality of the explanation.

Use one decimal place. No extra keys. No markdown.
"""


def clamp_score(value: float, low: float = 1.0, high: float = 5.0) -> float:
    return max(low, min(high, float(value)))


def to_percent(score_1_to_5: float) -> float:
    return round(float(score_1_to_5) * 20.0, 1)


def parse_json_block(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_scores(raw: Dict) -> Dict:
    fields = ["faithfulness", "personalization", "persuasiveness", "fluency"]
    scores = {}
    for key in fields:
        if key not in raw:
            raise ValueError(f"Missing key in model output: {key}")
        scores[key] = round(clamp_score(raw[key]), 1)

    overall = raw.get("overall")
    if overall is None:
        overall = sum(scores.values()) / len(scores)
    scores["overall"] = round(clamp_score(overall), 1)

    reason = str(raw.get("reason", "")).strip()
    scores["reason"] = reason
    return scores


def build_user_prompt(
    history: str,
    item: str,
    prediction: str,
    reference: Optional[str] = None,
) -> str:
    prompt = (
        "Evaluate the model explanation.\n\n"
        f"[User History]\n{history}\n\n"
        f"[Recommended Item]\n{item}\n\n"
        f"[Model Explanation]\n{prediction}\n"
    )
    if reference is not None:
        prompt += f"\n[Reference Explanation]\n{reference}\n"
    return prompt


def parse_history_item_from_prompt(source_text: str) -> Tuple[str, str]:
    history = source_text
    item = ""

    match = re.search(
        r"User History:\s*(.*?)\nRecommended Item:\s*(.*?)\nExplanation:",
        source_text,
        flags=re.DOTALL,
    )
    if match:
        history = match.group(1).strip()
        item = match.group(2).strip()
        return history, item

    if "User History:" in source_text:
        history = source_text.split("User History:", 1)[1].strip()
    if "Recommended Item:" in source_text:
        item = source_text.split("Recommended Item:", 1)[1].split("\n", 1)[0].strip()

    return history.strip(), item.strip()


class GPTScorer:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_retries: int,
        retry_wait: float,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_wait = retry_wait

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=key, base_url=base_url)

    def _call_model(self, user_prompt: str) -> str:
        # Try chat-completions with JSON object mode first.
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content or ""
            if content.strip():
                return content
        except Exception:
            pass

        # Fallback: same API without response_format.
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content or ""
            if content.strip():
                return content
        except Exception:
            pass

        # Final fallback: Responses API.
        resp = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        output_text = getattr(resp, "output_text", "")
        if not output_text:
            raise RuntimeError("Empty model response.")
        return output_text

    def score_one(
        self,
        history: str,
        item: str,
        prediction: str,
        reference: Optional[str] = None,
    ) -> Dict:
        prompt = build_user_prompt(history, item, prediction, reference)
        last_err = None

        for attempt in range(1, self.max_retries + 1):
            try:
                text = self._call_model(prompt)
                raw = parse_json_block(text)
                return normalize_scores(raw)
            except Exception as err:
                last_err = err
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait * attempt)

        raise RuntimeError(f"Failed after {self.max_retries} retries: {last_err}")


def build_predictions_from_model(args: argparse.Namespace) -> pd.DataFrame:
    import torch
    from argparse import Namespace
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    genrec_root = os.path.join(project_root, "GenRec")
    if genrec_root not in sys.path:
        sys.path.insert(0, genrec_root)

    from genrec.data import Dataset
    from genrec.model import GenerativeModel
    from genrec.utils import move_to_cuda

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_base_path,
        local_files_only=args.local_files_only,
        add_prefix_space=True,
    )
    tokenizer.add_tokens(["<mask>"])

    gen_config = Namespace(
        model_name=args.model_base_path,
        local_files_only=args.local_files_only,
        label_smoothing=args.label_smoothing,
    )

    model = GenerativeModel(gen_config, tokenizer)
    state_dict = torch.load(args.model_weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    test_set = Dataset(
        tokenizer,
        max_length=args.max_length,
        path=args.test_data_path,
        max_output_length=args.max_output_length,
        filter_pseudo_labels=False,
        history_max_chars=args.history_max_chars,
    )
    if len(test_set) == 0:
        raise ValueError("Test set is empty after loading.")

    test_loader = DataLoader(
        test_set,
        batch_size=args.gen_batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn,
        num_workers=args.gen_num_workers,
        pin_memory=(device.type == "cuda"),
    )

    rows = []
    stop = False
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating explanations"):
            gpu_batch = move_to_cuda(batch, device=device) if device.type == "cuda" else batch
            predictions = model.predict(
                gpu_batch,
                num_beams=args.eval_num_beams,
                max_length=args.max_output_length,
                repetition_penalty=args.eval_repetition_penalty,
                no_repeat_ngram_size=args.eval_no_repeat_ngram_size,
                length_penalty=args.eval_length_penalty,
                min_new_tokens=args.eval_min_new_tokens,
            )
            for src, refs, pred in zip(batch.input_text, batch.target_text, predictions):
                history, item = parse_history_item_from_prompt(str(src))
                ref = refs[0] if refs else ""
                rows.append(
                    {
                        "history": history,
                        "item": item,
                        "golden": ref,
                        "prediction": pred,
                    }
                )
                if args.max_samples > 0 and len(rows) >= args.max_samples:
                    stop = True
                    break
            if stop:
                break

    pred_df = pd.DataFrame(rows)
    if len(pred_df) == 0:
        raise ValueError("No prediction rows were generated.")

    if args.generated_csv:
        os.makedirs(os.path.dirname(args.generated_csv) or ".", exist_ok=True)
        pred_df.to_csv(args.generated_csv, index=False)
        print(f"[generate] saved raw predictions -> {args.generated_csv}")

    return pred_df


def load_predictions_from_csv(args: argparse.Namespace) -> pd.DataFrame:
    if not args.input_csv:
        raise ValueError("--input_csv is required when --mode from_csv")
    df = pd.read_csv(args.input_csv)
    if args.max_samples and args.max_samples > 0:
        df = df.head(args.max_samples).copy()
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate explanations (optional) and compute GPT-score."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="from_model",
        choices=["from_model", "from_csv"],
        help="from_model: generate with trained weights then score; from_csv: score existing CSV.",
    )

    parser.add_argument("--input_csv", type=str, default="", help="Input CSV for mode=from_csv.")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV with GPT-score columns.")
    parser.add_argument("--summary_json", type=str, required=True, help="Summary JSON path.")
    parser.add_argument(
        "--generated_csv",
        type=str,
        default="",
        help="Optional: save raw generated predictions (mode=from_model).",
    )

    # Required for mode=from_model
    parser.add_argument("--model_base_path", type=str, default="", help="Base model path (e.g. bart-base).")
    parser.add_argument("--model_weights_path", type=str, default="", help="Trained weights path (best_model.mdl).")
    parser.add_argument("--test_data_path", type=str, default="", help="Test CSV path.")

    parser.add_argument("--gen_batch_size", type=int, default=16, help="Generation batch size.")
    parser.add_argument("--gen_num_workers", type=int, default=0, help="DataLoader workers for generation.")
    parser.add_argument("--max_length", type=int, default=768, help="Encoder max length.")
    parser.add_argument("--max_output_length", type=int, default=150, help="Decoder max length.")
    parser.add_argument("--history_max_chars", type=int, default=3000, help="History truncation length.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Model label_smoothing value.")
    parser.add_argument("--eval_num_beams", type=int, default=5, help="Beam size for generation.")
    parser.add_argument("--eval_repetition_penalty", type=float, default=1.2, help="Repetition penalty.")
    parser.add_argument("--eval_no_repeat_ngram_size", type=int, default=3, help="No-repeat ngram size.")
    parser.add_argument("--eval_length_penalty", type=float, default=1.0, help="Length penalty.")
    parser.add_argument("--eval_min_new_tokens", type=int, default=0, help="Min new tokens.")

    parser.add_argument("--history_col", type=str, default="history", help="History column name.")
    parser.add_argument("--item_col", type=str, default="item", help="Item column name.")
    parser.add_argument("--prediction_col", type=str, default="prediction", help="Prediction column name.")
    parser.add_argument("--reference_col", type=str, default="golden", help="Reference column name.")
    parser.add_argument(
        "--use_reference",
        action="store_true",
        help="Whether to pass reference explanation to the judge prompt.",
    )

    # Default changed to GPT-3.5 as requested.
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Judge model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per sample.")
    parser.add_argument("--retry_wait", type=float, default=1.5, help="Base retry wait (seconds).")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep time between calls (seconds).")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--save_every", type=int, default=20, help="Save output every N newly scored rows.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file.")

    parser.add_argument("--api_key", type=str, default="", help="Optional API key override.")
    parser.add_argument("--base_url", type=str, default="", help="Optional OpenAI-compatible base URL.")

    parser.add_argument("--local_files_only", dest="local_files_only", action="store_true")
    parser.add_argument("--no_local_files_only", dest="local_files_only", action="store_false")
    parser.set_defaults(local_files_only=True)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "from_model":
        required = {
            "--model_base_path": args.model_base_path,
            "--model_weights_path": args.model_weights_path,
            "--test_data_path": args.test_data_path,
        }
        missing = [k for k, v in required.items() if not str(v).strip()]
        if missing:
            raise ValueError(f"Missing required args for --mode from_model: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.mode == "from_model":
        df = build_predictions_from_model(args)
    else:
        df = load_predictions_from_csv(args)

    required_cols = [args.history_col, args.item_col, args.prediction_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if args.use_reference and args.reference_col not in df.columns:
        raise ValueError(
            f"--use_reference is set but reference column does not exist: {args.reference_col}"
        )

    df = df.reset_index(drop=False).rename(columns={"index": "_row_id"})
    score_cols = [
        "gpt_faithfulness",
        "gpt_personalization",
        "gpt_persuasiveness",
        "gpt_fluency",
        "gpt_overall",
        "gpt_faithfulness_100",
        "gpt_personalization_100",
        "gpt_persuasiveness_100",
        "gpt_fluency_100",
        "gpt_overall_100",
        "gpt_score",
        "gpt_reason",
        "gpt_error",
    ]
    for col in score_cols:
        if col not in df.columns:
            df[col] = pd.NA

    if args.resume and os.path.exists(args.output_csv):
        prev = pd.read_csv(args.output_csv)
        if "_row_id" in prev.columns:
            prev = prev.set_index("_row_id")
            for idx in df["_row_id"].tolist():
                if idx in prev.index:
                    for col in score_cols:
                        if col in prev.columns:
                            df.loc[df["_row_id"] == idx, col] = prev.at[idx, col]

    scorer = GPTScorer(
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        retry_wait=args.retry_wait,
        api_key=args.api_key or None,
        base_url=args.base_url or None,
    )

    processed = 0
    total = len(df)
    for i, row in df.iterrows():
        if pd.notna(row["gpt_overall"]):
            continue

        history = str(row[args.history_col]) if pd.notna(row[args.history_col]) else ""
        item = str(row[args.item_col]) if pd.notna(row[args.item_col]) else ""
        prediction = str(row[args.prediction_col]) if pd.notna(row[args.prediction_col]) else ""
        reference = None
        if args.use_reference:
            reference = str(row[args.reference_col]) if pd.notna(row[args.reference_col]) else ""

        try:
            scores = scorer.score_one(
                history=history,
                item=item,
                prediction=prediction,
                reference=reference,
            )
            df.at[i, "gpt_faithfulness"] = scores["faithfulness"]
            df.at[i, "gpt_personalization"] = scores["personalization"]
            df.at[i, "gpt_persuasiveness"] = scores["persuasiveness"]
            df.at[i, "gpt_fluency"] = scores["fluency"]
            df.at[i, "gpt_overall"] = scores["overall"]
            df.at[i, "gpt_faithfulness_100"] = to_percent(scores["faithfulness"])
            df.at[i, "gpt_personalization_100"] = to_percent(scores["personalization"])
            df.at[i, "gpt_persuasiveness_100"] = to_percent(scores["persuasiveness"])
            df.at[i, "gpt_fluency_100"] = to_percent(scores["fluency"])
            df.at[i, "gpt_overall_100"] = to_percent(scores["overall"])
            df.at[i, "gpt_score"] = to_percent(scores["overall"])
            df.at[i, "gpt_reason"] = scores["reason"]
            df.at[i, "gpt_error"] = pd.NA
        except Exception as err:
            df.at[i, "gpt_error"] = str(err)

        processed += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

        if processed % args.save_every == 0:
            os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
            df.to_csv(args.output_csv, index=False)
            print(f"[progress] saved {processed} new rows, total rows={total}")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    # Backfill percentage columns for resumed files that only have 1-5 scores.
    pairs = [
        ("gpt_faithfulness", "gpt_faithfulness_100"),
        ("gpt_personalization", "gpt_personalization_100"),
        ("gpt_persuasiveness", "gpt_persuasiveness_100"),
        ("gpt_fluency", "gpt_fluency_100"),
        ("gpt_overall", "gpt_overall_100"),
    ]
    for base_col, pct_col in pairs:
        base_series = pd.to_numeric(df[base_col], errors="coerce")
        need_fill = df[pct_col].isna() & base_series.notna()
        df.loc[need_fill, pct_col] = (base_series[need_fill] * 20.0).round(1)
    score_series = pd.to_numeric(df["gpt_overall_100"], errors="coerce")
    need_score_fill = df["gpt_score"].isna() & score_series.notna()
    df.loc[need_score_fill, "gpt_score"] = score_series[need_score_fill]

    numeric_cols = [
        "gpt_faithfulness_100",
        "gpt_personalization_100",
        "gpt_persuasiveness_100",
        "gpt_fluency_100",
        "gpt_overall_100",
        "gpt_score",
    ]
    summary = {
        "mode": args.mode,
        "output_csv": args.output_csv,
        "judge_model": args.model,
        "scored_rows": int(df["gpt_score"].notna().sum()),
        "error_rows": int(df["gpt_error"].notna().sum()),
        "metrics": {},
    }
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        summary["metrics"][col] = {
            "mean": round(float(series.mean()), 4) if len(series) else None,
            "std": round(float(series.std(ddof=0)), 4) if len(series) else None,
            "count": int(len(series)),
        }

    os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nGPT-score finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
