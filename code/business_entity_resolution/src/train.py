import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Set, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalization import normalize_text
from src.blocking import generate_ultra_high_recall_blocking_candidates
from src.features import extract_pair_features
from src.evaluation import evaluate_predictions_macro_f05, verify_evaluator_against_worked_example
from src.classifier import EntityResolutionClassifier

def parse_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Loads train_ground_truth.tsv into a dictionary mapping s1_id -> set of matched_entity_ids."""
    gt_df = pd.read_csv(gt_path, sep='\t')
    gt_dict = {}
    for row in gt_df.itertuples():
        s1_id = row.source1_entity_id
        matched_str = str(row.matched_entity_ids) if pd.notna(row.matched_entity_ids) else ""
        matched_set = set(matched_str.split(',')) if matched_str.strip() else set()
        gt_dict[s1_id] = matched_set
    return gt_dict

def load_true_match_targets(dataset_dir: str, needed_ids: Set[str], bg_samples: int = 15000) -> pd.DataFrame:
    """Efficiently loads all ground-truth matching S2/S3 records plus background records."""
    train_dir = os.path.join(dataset_dir, 'train')
    
    s2_matches = []
    for chunk in pd.read_csv(os.path.join(train_dir, 'train_source2.tsv'), sep='\t', chunksize=500000):
        m = chunk[chunk['entity_id'].isin(needed_ids)]
        if len(m) > 0:
            s2_matches.append(m)
    s2_matched_df = pd.concat(s2_matches, ignore_index=True) if s2_matches else pd.DataFrame()
    
    s3_matches = []
    for chunk in pd.read_csv(os.path.join(train_dir, 'train_source3.tsv'), sep='\t', chunksize=500000):
        m = chunk[chunk['entity_id'].isin(needed_ids)]
        if len(m) > 0:
            s3_matches.append(m)
    s3_matched_df = pd.concat(s3_matches, ignore_index=True) if s3_matches else pd.DataFrame()

    # Add background samples for negative candidate pool
    s2_bg = pd.read_csv(os.path.join(train_dir, 'train_source2.tsv'), sep='\t', nrows=bg_samples)
    s3_bg = pd.read_csv(os.path.join(train_dir, 'train_source3.tsv'), sep='\t', nrows=bg_samples)

    targets = pd.concat([s2_matched_df, s3_matched_df, s2_bg, s3_bg], ignore_index=True).drop_duplicates(subset=['entity_id'])
    return targets

def train_and_evaluate(dataset_dir: str, sample_size: int = 2000):
    print("=== Training & Validation Protocol for Business Entity Resolution ===", flush=True)
    
    # 1. Verify Evaluator
    assert verify_evaluator_against_worked_example(), "Evaluator check failed!"
    print("[PASS] Local F_0.5 evaluator verified against PDF worked example.", flush=True)

    train_dir = os.path.join(dataset_dir, 'train')
    
    # 2. Load Train Files & Target Matches
    print(f"\n1. Loading training reference sample (sample_size={sample_size})...", flush=True)
    s1_train = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t', nrows=sample_size)
    gt_dict = parse_ground_truth(os.path.join(train_dir, 'train_ground_truth.tsv'))

    needed_ids = set()
    for s1_id in s1_train['entity_id']:
        needed_ids.update(gt_dict.get(s1_id, set()))

    print(f"Loading true matched S2/S3 target entities ({len(needed_ids)} target IDs required)...", flush=True)
    targets_df = load_true_match_targets(dataset_dir, needed_ids, bg_samples=sample_size * 5)
    print(f"Loaded target candidate pool: {len(targets_df)} total records.", flush=True)

    # 3. Entity-Grouped Train / Validation Split
    print("2. Creating entity-level train / validation split (80% train, 20% val)...", flush=True)
    s1_ids = s1_train['entity_id'].tolist()
    train_ids, val_ids = train_test_split(s1_ids, test_size=0.20, random_state=42)
    
    s1_train_df = s1_train[s1_train['entity_id'].isin(train_ids)].copy()
    s1_val_df = s1_train[s1_train['entity_id'].isin(val_ids)].copy()
    
    val_gt_dict = {s1_id: gt_dict.get(s1_id, set()) for s1_id in val_ids}

    # 4. Text Normalization
    print("3. Normalizing business names and addresses...", flush=True)
    for df in [s1_train_df, s1_val_df, targets_df]:
        df['business_name_clean'] = df['business_name'].apply(normalize_text)
        df['business_address_clean'] = df['business_address'].apply(normalize_text)

    # 5. Multi-Strategy Candidate Blocking & Recall Evaluation
    print("4. Generating candidate pairs via Ultra-High Recall Union Engine (Token + 3-Gram TF-IDF)...", flush=True)
    val_candidates = generate_ultra_high_recall_blocking_candidates(s1_val_df, targets_df, top_k_tfidf=30, max_candidates_cap=60)

    # Measure Candidate Blocking Recall Gate
    recalled_count = 0
    total_true_matches = 0
    for s1_id, true_set in val_gt_dict.items():
        total_true_matches += len(true_set)
        recalled_count += len(true_set.intersection(val_candidates.get(s1_id, set())))
        
    blocking_recall = (recalled_count / total_true_matches) if total_true_matches > 0 else 1.0
    print(f"[GATE 2] Validation Ultra-High Recall Blocking Recall = {blocking_recall:.4f} ({recalled_count}/{total_true_matches}) [{blocking_recall*100:.2f}%]", flush=True)

    # 6. Build Pairwise Training Samples with Hard Negatives
    print("5. Extracting 30-feature vector matrix & mining hard negatives...", flush=True)
    train_candidates = generate_ultra_high_recall_blocking_candidates(s1_train_df, targets_df, top_k_tfidf=30, max_candidates_cap=60)

    s1_dict = s1_train_df.set_index('entity_id').to_dict('index')
    target_dict = targets_df.set_index('entity_id').to_dict('index')

    feature_rows = []
    labels = []
    
    for s1_id in s1_train_df['entity_id']:
        true_set = gt_dict.get(s1_id, set())
        cands = train_candidates.get(s1_id, set())
        s1_row = s1_dict[s1_id]
        
        for cand_id in cands:
            if cand_id in target_dict:
                cand_row = target_dict[cand_id]
                feats = extract_pair_features(s1_row, cand_row)
                is_match = 1.0 if cand_id in true_set else 0.0
                
                feature_rows.append(feats)
                labels.append(is_match)

    X_train = pd.DataFrame(feature_rows)
    y_train = np.array(labels)
    print(f"  Extracted {len(X_train)} training candidate pairs (Positives: {int(y_train.sum())}, Negatives: {len(y_train) - int(y_train.sum())}).", flush=True)

    # 7. Model Training & Validation Threshold Optimization
    print("6. Training Calibrated GBDT Ensemble & fine-tuning threshold tau...", flush=True)
    clf = EntityResolutionClassifier(n_estimators=200)
    clf.fit(X_train, y_train)

    # Extract validation features
    val_s1_dict = s1_val_df.set_index('entity_id').to_dict('index')
    val_rows = []
    for s1_id in s1_val_df['entity_id']:
        cands = val_candidates.get(s1_id, set())
        s1_row = val_s1_dict[s1_id]
        for cand_id in cands:
            if cand_id in target_dict:
                cand_row = target_dict[cand_id]
                feats = extract_pair_features(s1_row, cand_row)
                feats['source1_entity_id'] = s1_id
                feats['candidate_entity_id'] = cand_id
                val_rows.append(feats)

    val_feats_df = pd.DataFrame(val_rows)
    best_val_f05, best_tau = clf.optimize_threshold_f05(val_feats_df, val_gt_dict)
    print(f"\n[RESULTS] Optimal Global Threshold tau = {best_tau:.2f}")
    print(f"[RESULTS] Validation Macro F_0.5 Score = {best_val_f05:.4f} ({best_val_f05*100:.2f}%)", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='6ab10eb3b23ba_student_resource/student_resource/dataset')
    parser.add_argument('--sample-size', type=int, default=2000)
    args = parser.parse_args()
    train_and_evaluate(args.dataset_dir, args.sample_size)
