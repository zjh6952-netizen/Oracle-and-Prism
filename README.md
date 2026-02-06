# Oracle & Prism

> Faithful and Personalized Recommendation Explanation via Knowledge Distillation

<!-- [[Paper]]() &nbsp; -->

## Overview

Large language models can produce high-quality recommendation explanations, but their inference cost makes direct deployment impractical. **Oracle & Prism** addresses this through a two-stage distillation pipeline:

1. **Oracle** (FLAN-T5-XXL, 11B params) generates candidate explanations conditioned on user interaction histories, then reranks and filters them to form pseudo-labels.
2. **Prism** (a user/item-aware BART-base, 140M params) learns from these pseudo-labels using a two-phase training strategy with discriminative learning rates.

The resulting student model achieves ~10x memory reduction and ~24x inference speedup over the teacher while preserving explanation quality. Prism is also recommender-agnostic — it can generate explanations for any upstream recommender in a plug-and-play fashion.

The key design choices are:
- A shared *faithful prompt template* between teacher and student that constrains generation to facts grounded in user history, reducing hallucination.
- A *fusion gate* that injects learnable user/item embeddings into the BART encoder, enabling personalized generation without retrieval augmentation.
- A *pseudo-label quality filter* based on source-target overlap, repetition ratio, and a composite quality score.

<!-- ![framework](assets/framework.png) -->

## Repository Structure

```
Oracle-and-Prism/
├── GenRec/
│   ├── config/                  # Training configs (JSON)
│   ├── genrec/
│   │   ├── train.py             # Main training loop
│   │   ├── model.py             # GenerativeModel wrapper
│   │   ├── data.py              # Dataset & quality filtering
│   │   ├── genrec_evaluate.py   # Dev-set evaluation
│   │   ├── utils.py
│   │   └── bart/
│   │       └── model.py         # Modified BART with user/item fusion gate
│   └── datasets/
├── scripts/
│   ├── create_explanation_dataset.py   # Oracle inference & pseudo-label generation
│   ├── preprocess_yelp.py
│   ├── preprocess_movielens.py
│   ├── evaluate_our_model.py           # ROUGE & BERTScore
│   ├── evaluate_gpt_score.py           # GPT-as-a-judge
│   ├── evaluate_baseline.py
│   ├── analyze_human_eval.py
│   ├── test_efficiency.py
│   └── train.sh
├── offline_metrics/
└── requirements.txt
```

## Getting Started

**Requirements:** Python 3.10+, CUDA GPU with BF16/FP16 support.

```bash
pip install -r requirements.txt
```

### 1. Data Preprocessing

```bash
python scripts/preprocess_yelp.py
# or
python scripts/preprocess_movielens.py
```

### 2. Pseudo-label Generation (Oracle)

```bash
python scripts/create_explanation_dataset.py
```

This produces `explanation_dataset_{train,test}.csv` with quality-filtered pseudo-labels.

### 3. Student Training (Prism)

```bash
python genrec/train.py -c config/genrec_e_movielens_config.json
```

Checkpoints, logs, and dev-set predictions are saved to a timestamped directory under `GenRec/results/`. Training configs are in `GenRec/config/`.

### 4. Evaluation

```bash
# Automatic metrics
python scripts/evaluate_our_model.py

# GPT-score (requires OpenAI API key)
python scripts/evaluate_gpt_score.py --mode from_model \
    --model_base_path /path/to/bart-base \
    --model_weights_path /path/to/best_model.mdl \
    --test_data_path /path/to/test.csv \
    --output_csv results/gpt_scores.csv \
    --summary_json results/gpt_summary.json
```

## Datasets

We evaluate on **Yelp** (business recommendations) and **MovieLens** (movie recommendations). The expected CSV format has four columns: `history`, `recommended_item`, `user_id`, `explanation`.

## Evaluation Metrics

| Metric | Description |
|---|---|
| ROUGE-L | N-gram overlap with reference explanations |
| BERTScore | Semantic similarity via contextual embeddings |
| GPT-Score | LLM-as-a-judge on faithfulness, personalization, persuasiveness, and fluency (0-100) |
| Human Eval | Annotator ratings on faithfulness, informativeness, and fluency |

<!--
## Citation

```bibtex
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={}
}
```
-->
