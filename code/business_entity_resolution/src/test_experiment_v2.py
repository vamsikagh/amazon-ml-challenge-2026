import os
import sys
import gc
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import fuzz, distance
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalization import normalize_text, extract_pin_code, extract_door_numbers
from src.evaluation import evaluate_predictions_macro_f05, compute_entity_f05

def parse_ground_truth(gt_path: str) -> dict:
    gt_df = pd.read_csv(gt_path, sep='\t')
    gt_dict = {}
    for row in gt_df.itertuples():
        s1_id = row.source1_entity_id
        matched_str = str(row.matched_entity_ids) if pd.notna(row.matched_entity_ids) else ""
        matched_set = set(matched_str.split(',')) if matched_str.strip() else set()
        gt_dict[s1_id] = matched_set
    return gt_dict

def load_true_match_targets(dataset_dir: str, needed_ids: set, bg_samples: int = 30000) -> pd.DataFrame:
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

    s2_bg = pd.read_csv(os.path.join(train_dir, 'train_source2.tsv'), sep='\t', nrows=bg_samples)
    s3_bg = pd.read_csv(os.path.join(train_dir, 'train_source3.tsv'), sep='\t', nrows=bg_samples)

    targets = pd.concat([s2_matched_df, s3_matched_df, s2_bg, s3_bg], ignore_index=True).drop_duplicates(subset=['entity_id'])
    return targets

def generate_multi_strategy_candidates_v2(s1_df, target_df, top_k=50):
    s1_ids = s1_df['entity_id'].values
    target_ids = target_df['entity_id'].values
    
    s1_texts = (s1_df['business_name_clean'].fillna('') + ' ' + s1_df['business_address_clean'].fillna('')).tolist()
    target_texts = (target_df['business_name_clean'].fillna('') + ' ' + target_df['business_address_clean'].fillna('')).tolist()
    
    # 1. Combo Name + Addr TF-IDF
    vec_combo = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=150000)
    target_vecs_combo = vec_combo.fit_transform(target_texts)
    nn_combo = NearestNeighbors(n_neighbors=min(top_k, len(target_texts)), metric='cosine', algorithm='brute')
    nn_combo.fit(target_vecs_combo)
    
    s1_vecs_combo = vec_combo.transform(s1_texts)
    _, combo_indices = nn_combo.kneighbors(s1_vecs_combo)
    
    # 2. Inverted Index for Name Tokens, Address Tokens, and PINs
    target_token_index = defaultdict(list)
    target_pin_index = defaultdict(list)
    
    GENERIC_STOP = {'private', 'limited', 'incorporated', 'corporation', 'company', 'india', 'services', 'street', 'road', 'building', 'floor', 'suite'}
    
    for idx, row in enumerate(target_df.itertuples()):
        name = getattr(row, 'business_name_clean', '')
        addr = getattr(row, 'business_address_clean', '')
        
        # Name + Address tokens
        tokens = set([t for t in (name + ' ' + addr).split() if len(t) >= 4 and t not in GENERIC_STOP])
        for t in tokens:
            target_token_index[t].append(idx)
            
        pin = extract_pin_code(addr)
        if pin:
            target_pin_index[pin].append(idx)
            
    candidates = defaultdict(set)
    for idx_offset, sid in enumerate(s1_ids):
        cands = set(target_ids[combo_indices[idx_offset]])
        
        name = s1_df.iloc[idx_offset]['business_name_clean']
        addr = s1_df.iloc[idx_offset]['business_address_clean']
        
        tokens = set([t for t in (name + ' ' + addr).split() if len(t) >= 4 and t not in GENERIC_STOP])
        for t in tokens:
            if t in target_token_index and len(target_token_index[t]) <= 600:
                cands.update([target_ids[i] for i in target_token_index[t][:40]])
                
        pin = extract_pin_code(addr)
        if pin and pin in target_pin_index and len(target_pin_index[pin]) <= 600:
            cands.update([target_ids[i] for i in target_pin_index[pin][:40]])
            
        candidates[sid] = cands
        
    return candidates

def extract_features_v2(s1_row, cand_row):
    s1_raw_name = str(s1_row.get('business_name', ''))
    cand_raw_name = str(cand_row.get('business_name', ''))
    
    s1_name = str(s1_row.get('business_name_clean', ''))
    cand_name = str(cand_row.get('business_name_clean', ''))
    s1_addr = str(s1_row.get('business_address_clean', ''))
    cand_addr = str(cand_row.get('business_address_clean', ''))
    s1_country = str(s1_row.get('country', '')).lower()
    cand_country = str(cand_row.get('country', '')).lower()
    
    # 1. Non-ASCII / Transliteration Flags
    s1_has_non_ascii = 1.0 if any(ord(c) > 127 for c in s1_raw_name) else 0.0
    cand_has_non_ascii = 1.0 if any(ord(c) > 127 for c in cand_raw_name) else 0.0
    s1_name_empty = 1.0 if not s1_name.strip() else 0.0
    cand_name_empty = 1.0 if not cand_name.strip() else 0.0
    
    # 2. Name Similarities
    name_lev = distance.Levenshtein.normalized_similarity(s1_name, cand_name) if s1_name and cand_name else 0.0
    name_jw = distance.JaroWinkler.similarity(s1_name, cand_name) if s1_name and cand_name else 0.0
    name_tsort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_tset = fuzz.token_set_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_ratio = fuzz.ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    
    w1_s1 = s1_name.split()[0] if s1_name else ""
    w1_cand = cand_name.split()[0] if cand_name else ""
    first_word_match = 1.0 if (w1_s1 and w1_cand and w1_s1 == w1_cand) else 0.0
    
    # 3. Address Similarities
    has_addr = bool(s1_addr and cand_addr)
    addr_lev = distance.Levenshtein.normalized_similarity(s1_addr, cand_addr) if has_addr else 0.0
    addr_jw = distance.JaroWinkler.similarity(s1_addr, cand_addr) if has_addr else 0.0
    addr_tsort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0 if has_addr else 0.0
    addr_tset = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0 if has_addr else 0.0
    addr_partial = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0 if has_addr else 0.0
    
    s1_pin = extract_pin_code(s1_addr)
    cand_pin = extract_pin_code(cand_addr)
    pin_match = 1.0 if (s1_pin and cand_pin and s1_pin == cand_pin) else 0.0
    pin_mismatch = 1.0 if (s1_pin and cand_pin and s1_pin != cand_pin) else 0.0
    
    s1_nums = extract_door_numbers(s1_addr)
    cand_nums = extract_door_numbers(cand_addr)
    num_match = 1.0 if (s1_nums and cand_nums and len(s1_nums.intersection(cand_nums)) > 0) else 0.0
    num_mismatch = 1.0 if (s1_nums and cand_nums and len(s1_nums.intersection(cand_nums)) == 0) else 0.0
    
    country_match = 1.0 if (s1_country and cand_country and s1_country == cand_country) else 0.0
    country_mismatch = 1.0 if (s1_country and cand_country and s1_country != cand_country) else 0.0
    
    # 4. Composite & High-Precision Interactions
    max_sim = max(name_tset, addr_tset)
    min_sim = min(name_tset, addr_tset)
    name_addr_prod = name_tset * addr_tset
    
    # Address Dominant Fallback (Handles non-ASCII names or missing name matches)
    addr_dominant = addr_tset if (s1_name_empty or cand_name_empty or s1_has_non_ascii or cand_has_non_ascii) else 0.0

    return {
        'name_lev': name_lev, 'name_jw': name_jw, 'name_tsort': name_tsort,
        'name_tset': name_tset, 'name_partial': name_partial, 'name_ratio': name_ratio,
        'first_word_match': first_word_match, 'addr_lev': addr_lev, 'addr_jw': addr_jw,
        'addr_tsort': addr_tsort, 'addr_tset': addr_tset, 'addr_partial': addr_partial,
        'pin_match': pin_match, 'pin_mismatch': pin_mismatch, 'num_match': num_match,
        'num_mismatch': num_mismatch, 'country_match': country_match, 'country_mismatch': country_mismatch,
        'has_addr': 1.0 if has_addr else 0.0,
        's1_has_non_ascii': s1_has_non_ascii, 'cand_has_non_ascii': cand_has_non_ascii,
        's1_name_empty': s1_name_empty, 'cand_name_empty': cand_name_empty,
        'max_sim': max_sim, 'min_sim': min_sim, 'name_addr_prod': name_addr_prod,
        'addr_dominant': addr_dominant
    }

def main():
    dataset_dir = '6ab10eb3b23ba_student_resource/student_resource/dataset'
    sample_size = 10000
    train_dir = os.path.join(dataset_dir, 'train')
    
    print(f"1. Loading training sample (size={sample_size})...")
    s1_df = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t', nrows=sample_size)
    gt_dict = parse_ground_truth(os.path.join(train_dir, 'train_ground_truth.tsv'))
    
    needed_ids = set()
    for sid in s1_df['entity_id']:
        needed_ids.update(gt_dict.get(sid, set()))
        
    print(f"Loading true target matches ({len(needed_ids)} target IDs required)...")
    targets_df = load_true_match_targets(dataset_dir, needed_ids, bg_samples=35000)
    print(f"Loaded target candidate pool: {len(targets_df)} total records.")

    print("2. Normalizing text...")
    for df in [s1_df, targets_df]:
        df['business_name_clean'] = df['business_name'].apply(normalize_text)
        df['business_address_clean'] = df['business_address'].apply(normalize_text)

    s1_ids = s1_df['entity_id'].tolist()
    train_ids, val_ids = train_test_split(s1_ids, test_size=0.20, random_state=42)
    
    s1_train_df = s1_df[s1_df['entity_id'].isin(train_ids)].copy()
    s1_val_df = s1_df[s1_df['entity_id'].isin(val_ids)].copy()
    val_gt_dict = {sid: gt_dict.get(sid, set()) for sid in val_ids}

    print("3. Generating candidate pools (v2)...")
    val_candidates = generate_multi_strategy_candidates_v2(s1_val_df, targets_df, top_k=50)
    
    recalled_count = 0
    total_true_matches = 0
    for sid, true_set in val_gt_dict.items():
        total_true_matches += len(true_set)
        recalled_count += len(true_set.intersection(val_candidates.get(sid, set())))
    blocking_recall = recalled_count / total_true_matches if total_true_matches > 0 else 1.0
    print(f"[GATE 2] Validation Candidate Blocking Recall = {blocking_recall:.4f} ({recalled_count}/{total_true_matches}) [{blocking_recall*100:.2f}%]")

    train_candidates = generate_multi_strategy_candidates_v2(s1_train_df, targets_df, top_k=50)
    
    s1_train_dict = s1_train_df.set_index('entity_id').to_dict('index')
    val_s1_dict = s1_val_df.set_index('entity_id').to_dict('index')
    target_dict = targets_df.set_index('entity_id').to_dict('index')

    print("4. Extracting feature vectors...")
    X_train_rows, y_train_list = [], []
    for sid in s1_train_df['entity_id']:
        true_set = gt_dict.get(sid, set())
        cands = train_candidates.get(sid, set())
        s1_row = s1_train_dict[sid]
        for cid in cands:
            if cid in target_dict:
                cand_row = target_dict[cid]
                feats = extract_features_v2(s1_row, cand_row)
                is_match = 1.0 if cid in true_set else 0.0
                X_train_rows.append(feats)
                y_train_list.append(is_match)
                
    X_train = pd.DataFrame(X_train_rows)
    y_train = np.array(y_train_list)
    print(f"  Training set: {len(X_train)} samples, Positives: {int(y_train.sum())}, Negatives: {len(y_train)-int(y_train.sum())}")

    print("5. Training Calibrated LightGBM Ensemble...")
    base_lgbm = LGBMClassifier(n_estimators=350, learning_rate=0.03, max_depth=9, num_leaves=127, random_state=42, n_jobs=-1, verbosity=-1)
    clf = CalibratedClassifierCV(estimator=base_lgbm, cv=3, method='isotonic')
    clf.fit(X_train, y_train)

    print("6. Sweeping optimal tau on validation set...")
    val_rows = []
    for sid in s1_val_df['entity_id']:
        cands = val_candidates.get(sid, set())
        s1_row = val_s1_dict[sid]
        for cid in cands:
            if cid in target_dict:
                cand_row = target_dict[cid]
                feats = extract_features_v2(s1_row, cand_row)
                feats['source1_entity_id'] = sid
                feats['candidate_entity_id'] = cid
                val_rows.append(feats)

    val_df = pd.DataFrame(val_rows)
    feature_cols = [c for c in val_df.columns if c not in ['source1_entity_id', 'candidate_entity_id']]
    probs = clf.predict_proba(val_df[feature_cols])[:, 1]
    val_df['prob'] = probs

    best_score = -1.0
    best_tau = 0.50

    for tau in np.arange(0.15, 0.90, 0.005):
        preds = defaultdict(set)
        matched_df = val_df[val_df['prob'] >= tau]
        for row in matched_df.itertuples():
            preds[row.source1_entity_id].add(row.candidate_entity_id)
            
        score = evaluate_predictions_macro_f05(val_gt_dict, preds)
        if score > best_score:
            best_score = score
            best_tau = tau

    print(f"\n==========================================")
    print(f"Optimal Global Threshold tau: {best_tau:.3f}")
    print(f"VALIDATION MACRO F_0.5 SCORE: {best_score:.4f} ({best_score*100:.2f}%)")
    print(f"==========================================")

if __name__ == '__main__':
    main()
