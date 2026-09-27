import os
import sys
import gc
import csv
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
from src.classifier import EntityResolutionClassifier

def parse_gt(path: str) -> Dict[str, Set[str]]:
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r, None)
        for row in r:
            m = set(x for x in (row[1] if len(row) > 1 else "").split(',') if x)
            d[row[0]] = m
    return d

def load_train_targets(needed_ids: Set[str], bg_samples: int, train_dir: str) -> pd.DataFrame:
    parts = []
    for src_file in ('train_source2.tsv', 'train_source3.tsv'):
        src_path = os.path.join(train_dir, src_file)
        if os.path.exists(src_path):
            for chunk in pd.read_csv(src_path, sep='\t', chunksize=500000):
                m = chunk[chunk['entity_id'].isin(needed_ids)]
                if len(m):
                    parts.append(m)
            parts.append(pd.read_csv(src_path, sep='\t', nrows=bg_samples))
    if not parts:
        raise FileNotFoundError(f"No train target files found in {train_dir}")
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=['entity_id'])

def run_pipeline(dataset_dir: str, output_dir: str, sample_size: int = None, train_samples: int = 15000):
    print("=== Amazon ML Challenge 2026 — High-Recall Multi-Match Pipeline (<4GB RAM) ===", flush=True)
    
    # 0. Verify local F_0.5 evaluator against PDF worked example
    if not verify_evaluator_against_worked_example():
        raise RuntimeError("Local F_0.5 evaluator verification failed against PDF worked example!")
    print("[PASS] Evaluator verified against PDF worked example.", flush=True)

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Train GBDT Classifier & Tune Decision Boundary Threshold tau*
    print("\n1. Training GBDT Classifier & calibrating threshold on training data...", flush=True)
    gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')
    s1_tr_path = os.path.join(train_dir, 'train_source1.tsv')
    
    gt = parse_gt(gt_path)
    s1_tr = pd.read_csv(s1_tr_path, sep='\t', nrows=train_samples)
    needed = set()
    for sid in s1_tr['entity_id']:
        needed |= gt.get(sid, set())
        
    targets_tr = load_train_targets(needed, bg_samples=train_samples * 5, train_dir=train_dir)
    print(f"  Loaded {len(s1_tr)} train S1 entities and {len(targets_tr)} train target entities.", flush=True)

    for df in (s1_tr, targets_tr):
        df['business_name_clean'] = df['business_name'].apply(normalize_text)
        df['business_address_clean'] = df['business_address'].apply(normalize_text)

    from sklearn.model_selection import train_test_split
    tr_ids, val_ids = train_test_split(s1_tr['entity_id'].tolist(), test_size=0.2, random_state=42)
    s1_sub_tr = s1_tr[s1_tr['entity_id'].isin(tr_ids)].copy()
    s1_sub_val = s1_tr[s1_tr['entity_id'].isin(val_ids)].copy()
    val_gt = {i: gt.get(i, set()) for i in val_ids}

    print("  Generating blocking candidates for training split...", flush=True)
    tr_cand = generate_ultra_high_recall_blocking_candidates(s1_sub_tr, targets_tr, top_k_tfidf=30, max_candidates_cap=60)
    val_cand = generate_ultra_high_recall_blocking_candidates(s1_sub_val, targets_tr, top_k_tfidf=30, max_candidates_cap=60)

    tname = dict(zip(targets_tr['entity_id'], targets_tr['business_name_clean']))
    taddr = dict(zip(targets_tr['entity_id'], targets_tr['business_address_clean']))
    tctry = dict(zip(targets_tr['entity_id'], targets_tr['country'].fillna('').astype(str)))

    def trow(i): return {'business_name_clean': tname[i], 'business_address_clean': taddr[i], 'country': tctry[i]}
    def srow(r): return {'business_name_clean': r.business_name_clean, 'business_address_clean': r.business_address_clean, 'country': str(r.country)}

    print("  Extracting 24 pair features for GBDT training...", flush=True)
    feats, labels = [], []
    s1tr_idx = {r.entity_id: srow(r) for r in s1_sub_tr.itertuples()}
    for sid in s1_sub_tr['entity_id']:
        tset = gt.get(sid, set()); sr = s1tr_idx[sid]
        for cid in tr_cand.get(sid, set()):
            if cid in tname:
                feats.append(extract_pair_features(sr, trow(cid)))
                labels.append(1.0 if cid in tset else 0.0)

    X_tr = pd.DataFrame(feats)
    y_tr = np.array(labels)
    print(f"  Training matrix: {len(X_tr)} candidate pairs ({int(y_tr.sum())} positive matches). Fitting LightGBM...", flush=True)
    
    clf = EntityResolutionClassifier(n_estimators=300)
    clf.fit(X_tr, y_tr)
    feat_cols = list(X_tr.columns)

    print("  Optimizing threshold tau* on held-out validation set...", flush=True)
    s1val_idx = {r.entity_id: srow(r) for r in s1_sub_val.itertuples()}
    vrows = []
    for sid in s1_sub_val['entity_id']:
        sr = s1val_idx[sid]
        for cid in val_cand.get(sid, set()):
            if cid in tname:
                f = extract_pair_features(sr, trow(cid))
                f['source1_entity_id'] = sid
                f['candidate_entity_id'] = cid
                vrows.append(f)
    vdf = pd.DataFrame(vrows)
    best_f05, best_tau = clf.optimize_threshold_f05(vdf, val_gt)
    print(f"  [TRAINED] Best Validation Macro F_0.5 = {best_f05:.4f} at tau* = {best_tau:.3f}", flush=True)

    del targets_tr, s1_tr, s1_sub_tr, s1_sub_val, X_tr, y_tr, vdf
    gc.collect()

    # 2. Test Set Candidate Generation & Inference
    print("\n2. Loading test set...", flush=True)
    s1_test = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t', nrows=sample_size)
    s2_test = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t', nrows=sample_size * 3 if sample_size else None)
    s3_test = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t', nrows=sample_size * 3 if sample_size else None)
    print(f"  Loaded S1 test: {len(s1_test)} rows, S2 test: {len(s2_test)} rows, S3 test: {len(s3_test)} rows.", flush=True)

    print("  Normalizing test text (unidecode transliteration + legal/address expansion)...", flush=True)
    for df in (s1_test, s2_test, s3_test):
        df['business_name_clean'] = df['business_name'].apply(normalize_text)
        df['business_address_clean'] = df['business_address'].apply(normalize_text)

    targets_test = pd.concat([s2_test, s3_test], ignore_index=True)
    del s2_test, s3_test
    gc.collect()

    # 3. Generating candidate_pairs.tsv
    print("\n3. Generating candidate pairs via sharded NMSLIB HNSW + Address Token Blocking...", flush=True)
    candidates_test = generate_ultra_high_recall_blocking_candidates(s1_test, targets_test, top_k_tfidf=30, max_candidates_cap=60)

    candidate_pairs_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    save_candidate_pairs_tsv(candidates_test, s1_test['entity_id'].tolist(), candidate_pairs_path)
    print(f"[SAVED] {candidate_pairs_path}", flush=True)

    # 4. Multi-Match Scoring for matching_results.tsv
    print(f"\n4. Scoring test candidate pairs with LightGBM (multi-match, tau*={best_tau:.3f})...", flush=True)
    test_tname = dict(zip(targets_test['entity_id'], targets_test['business_name_clean'].fillna('')))
    test_taddr = dict(zip(targets_test['entity_id'], targets_test['business_address_clean'].fillna('')))
    test_tctry = dict(zip(targets_test['entity_id'], targets_test['country'].fillna('').astype(str)))
    del targets_test
    gc.collect()

    matching_results_path = os.path.join(output_dir, 'matching_results.tsv')
    fout = open(matching_results_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(fout, delimiter='\t')
    writer.writerow(['source1_entity_id', 'matched_entity_ids'])

    matched_count = 0
    total_entities = len(s1_test)

    for s1_row in s1_test.itertuples():
        sid = s1_row.entity_id
        sr = {
            'business_name_clean': s1_row.business_name_clean,
            'business_address_clean': s1_row.business_address_clean,
            'country': str(s1_row.country)
        }
        cands = candidates_test.get(sid, set())
        matched_cands = []
        if cands:
            pair_feats = []
            cand_list = []
            for cid in cands:
                if cid in test_tname:
                    cr = {
                        'business_name_clean': test_tname[cid],
                        'business_address_clean': test_taddr[cid],
                        'country': test_tctry[cid]
                    }
                    pair_feats.append(extract_pair_features(sr, cr))
                    cand_list.append(cid)
            if pair_feats:
                X_pair = pd.DataFrame(pair_feats)[feat_cols]
                probs = clf.predict_proba(X_pair)
                for cid, p in zip(cand_list, probs):
                    if p >= best_tau:
                        matched_cands.append(cid)

        if matched_cands:
            matched_count += 1
            writer.writerow([sid, ",".join(sorted(matched_cands))])
        else:
            writer.writerow([sid, ""])

    fout.close()
    print(f"[SAVED] {matching_results_path}", flush=True)
    print(f"  Matched {matched_count}/{total_entities} S1 entities ({100.0 * matched_count / total_entities:.1f}% match rate).", flush=True)
    print("\n[COMPLETE] Pipeline finished successfully!", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='6ab10eb3b23ba_student_resource/student_resource/dataset')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--sample-size', type=int, default=None)
    parser.add_argument('--train-samples', type=int, default=15000)
    args = parser.parse_args()
    run_pipeline(args.dataset_dir, args.output_dir, args.sample_size, args.train_samples)
