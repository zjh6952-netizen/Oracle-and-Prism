#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""
Quick BERTScore test script.
Reads existing predictions from CSV and computes BERTScore.
Avoids reloading large generation models to save memory.
"""

import os
import sys
import pandas as pd
import evaluate

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CSV_FILE = os.path.join(RESULTS_DIR, 'evaluation_results_our_model.csv')
BERTSCORE_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'offline_metrics', 'bertscore')

NUM_SAMPLES = 100

def main():
    print("="*60)
    print("BERTScore Quick Test")
    print("="*60)
    
    if not os.path.exists(CSV_FILE):
        print(f" Error: CSV file not found: {CSV_FILE}")
        sys.exit(1)
    
    print(f"Found CSV file: {CSV_FILE}")
    
    print(f"\nReading CSV file...")
    df = pd.read_csv(CSV_FILE)
    print(f"CSV contains {len(df)} rows")
    
    df = df.head(NUM_SAMPLES)
    print(f"Running quick evaluation on the first {NUM_SAMPLES} rows")
    
    predictions = df['prediction'].fillna('').tolist()
    references = df['golden'].fillna('').tolist()
    
    valid_pairs = sum(1 for p, r in zip(predictions, references) if p.strip() and r.strip())
    print(f"Valid prediction-reference pairs: {valid_pairs}/{len(predictions)}")
    
    print(f"\nLoading BERTScore evaluator...")
    print(f"  - Offline script path: {BERTSCORE_SCRIPT_PATH}")
    
    try:
        bertscore = evaluate.load(BERTSCORE_SCRIPT_PATH)
        print("BERTScore evaluator loaded successfully")
    except Exception as e:
        print(f" Failed to load BERTScore: {e}")
        sys.exit(1)
    
    print(f"\nComputing BERTScore...")
    print(f"  - This may take a few minutes...")
    
    try:
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang='en',
            device='cpu',
            verbose=True
        )
        print("BERTScore computation finished")
    except Exception as e:
        print(f" BERTScore computation failed: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("BERTScore Evaluation Results")
    print("="*60)
    
    if 'precision' in results:
        avg_precision = sum(results['precision']) / len(results['precision'])
        print(f"Precision: {avg_precision:.4f}")
    
    if 'recall' in results:
        avg_recall = sum(results['recall']) / len(results['recall'])
        print(f"Recall:    {avg_recall:.4f}")
    
    if 'f1' in results:
        avg_f1 = sum(results['f1']) / len(results['f1'])
        print(f"F1 Score:  {avg_f1:.4f}")
    
    print("="*60)
    print(f"\nQuick test complete. Evaluated {NUM_SAMPLES} rows.")
    print(f"To evaluate all {len(pd.read_csv(CSV_FILE))} rows, update NUM_SAMPLES in this script.")
    
if __name__ == '__main__':
    main()
