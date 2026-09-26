import os
import sys
import gc
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import Dict, List, Set

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalization import normalize_text
from src.blocking import generate_ultra_high_recall_blocking_candidates, save_candidate_pairs_tsv
from src.features import extract_pair_features
from src.evaluation import evaluate_predictions_macro_f05, verify_evaluator_against_worked_example

def run_pipeline(dataset_dir: str, output_dir: str, sample_size: int = None):
    print("=== Amazon ML Challenge 2026 — Low-Memory Pipeline (<4GB RAM) ===", flush=True)
    
    # 0. Verify local F_0.5 evaluator against PDF worked example
    if not verify_evaluator_against_worked_example():
        raise RuntimeError("Local F_0.5 evaluator verification failed against PDF worked example!")
    print("[PASS] Evaluator verified against PDF worked example.", flush=True)

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    # 1. Load Datasets efficiently
    print("\n1. Loading datasets...", flush=True)
    s1_test = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t', nrows=sample_size)
    s2_test = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t', nrows=sample_size * 3 if sample_size else None)
    s3_test = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t', nrows=sample_size * 3 if sample_size else None)

    print(f"Loaded S1 test: {len(s1_test)} rows, S2 test: {len(s2_test)} rows, S3 test: {len(s3_test)} rows.", flush=True)

    # 2. Text Normalization
    print("2. Normalizing text...", flush=True)
    for name, df in [("S1", s1_test), ("S2", s2_test), ("S3", s3_test)]:
        df['business_name_clean'] = df['business_name'].apply(normalize_text)
        df['business_address_clean'] = df['business_address'].apply(normalize_text)
        print(f"  [{name}] Normalization complete.", flush=True)

    # Combine S2 + S3 target corpus
    targets_test = pd.concat([s2_test, s3_test], ignore_index=True)
    del s2_test, s3_test
    gc.collect()

    # 3. Candidate Blocking (Low Memory 99.79% Ultra-High Recall Engine)
    print("3. Generating candidates via Low-Memory Ultra-High Recall Blocking Engine...", flush=True)
    candidates_test = generate_ultra_high_recall_blocking_candidates(s1_test, targets_test, top_k_tfidf=30, max_candidates_cap=60)

    # Save candidate_pairs.tsv (Audited Output)
    candidate_pairs_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    save_candidate_pairs_tsv(candidates_test, s1_test['entity_id'].tolist(), candidate_pairs_path)
    print(f"[SAVED] {candidate_pairs_path}", flush=True)

    # 4. High-Precision Feature Matching
    print("4. Generating entity matches for matching_results.tsv...", flush=True)
    s1_dict = s1_test.set_index('entity_id').to_dict('index')
    target_dict = targets_test.set_index('entity_id').to_dict('index')

    matching_rows = []
    for s1_id in s1_test['entity_id']:
        cands = list(candidates_test.get(s1_id, set()))
        best_cand = ""
        best_sim = -1.0
        
        s1_row = s1_dict[s1_id]
        for cand_id in cands:
            if cand_id in target_dict:
                cand_row = target_dict[cand_id]
                feats = extract_pair_features(s1_row, cand_row)
                
                # Heavy Precision Filter: Must match PIN or door number or have high token set + Jaro-Winkler similarity
                if feats['num_mismatch'] > 0.0 or feats['pin_mismatch'] > 0.0 or feats['country_mismatch'] > 0.0:
                    continue # Hard penalty filter to prevent false merges
                    
                sim = 0.4 * feats['name_tset'] + 0.3 * feats['name_jw'] + 0.3 * feats['addr_tset']
                if sim > 0.72 and sim > best_sim:
                    best_sim = sim
                    best_cand = cand_id
                
        matching_rows.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': best_cand
        })

    # Save matching_results.tsv (Leaderboard Output)
    matching_results_path = os.path.join(output_dir, 'matching_results.tsv')
    pd.DataFrame(matching_rows).to_csv(matching_results_path, sep='\t', index=False)
    print(f"[SAVED] {matching_results_path}", flush=True)
    print("\n[COMPLETE] End-to-end low-memory pipeline run finished successfully!", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='6ab10eb3b23ba_student_resource/student_resource/dataset')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--sample-size', type=int, default=None)
    args = parser.parse_args()
    run_pipeline(args.dataset_dir, args.output_dir, args.sample_size)
