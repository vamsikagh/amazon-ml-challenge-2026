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

def load_true_match_targets(dataset_dir: str, needed_ids: set, bg_samples: int = 20000) -> pd.DataFrame:
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

def generate_multi_strategy_candidates(s1_df, target_df, top_k=50):
    s1_ids = s1_df['entity_id'].values
    target_ids = target_df['entity_id'].values
    
    s1_texts = (s1_df['business_name_clean'].fillna('') + ' ' + s1_df['business_address_clean'].fillna('')).tolist()
    target_texts = (target_df['business_name_clean'].fillna('') + ' ' + target_df['business_address_clean'].fillna('')).tolist()
    
    vec_combo = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_features=150000)
    target_vecs_combo = vec_combo.fit_transform(target_texts)
    nn_combo = NearestNeighbors(n_neighbors=min(top_k, len(target_texts)), metric='cosine', algorithm='brute')
    nn_combo.fit(target_vecs_combo)
    
    s1_vecs_combo = vec_combo.transform(s1_texts)
    _, combo_indices = nn_combo.kneighbors(s1_vecs_combo)
    
    target_name_index = defaultdict(list)
    target_pin_index = defaultdict(list)
    
    for idx, row in enumerate(target_df.itertuples()):
        name = row.business_name_clean
        addr = row.business_address_clean
        for t in name.split():
            if len(t) >= 4 and t not in {'private', 'limited', 'incorporated', 'corporation', 'company', 'india', 'services'}:
                target_name_index[t].append(idx)
        pin = extract_pin_code(addr)
        if pin:
            target_pin_index[pin].append(idx)
            
    candidates = defaultdict(set)
    for idx_offset, sid in enumerate(s1_df['entity_id'].values):
        cands = set(target_ids[combo_indices[idx_offset]])
        
        name = s1_df.iloc[idx_offset]['business_name_clean']
        addr = s1_df.iloc[idx_offset]['business_address_clean']
        
        for t in name.split():
            if len(t) >= 4 and t in target_name_index and len(target_name_index[t]) <= 500:
                cands.update([target_ids[i] for i in target_name_index[t][:50]])
                
        pin = extract_pin_code(addr)
        if pin and pin in target_pin_index and len(target_pin_index[pin]) <= 500:
            cands.update([target_ids[i] for i in target_pin_index[pin][:50]])
            
        candidates[sid] = cands
        
    return candidates

def extract_features(s1_row, cand_row):
    s1_name = str(s1_row.get('business_name_clean', ''))
    cand_name = str(cand_row.get('business_name_clean', ''))
    s1_addr = str(s1_row.get('business_address_clean', ''))
    cand_addr = str(cand_row.get('business_address_clean', ''))
    s1_country = str(s1_row.get('country', '')).lower()
    cand_country = str(cand_row.get('country', '')).lower()
    
    # 1. Name Similarities
    name_lev = distance.Levenshtein.normalized_similarity(s1_name, cand_name) if s1_name and cand_name else 0.0
    name_jw = distance.JaroWinkler.similarity(s1_name, cand_name) if s1_name and cand_name else 0.0
    name_tsort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_tset = fuzz.token_set_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_ratio = fuzz.ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    
    w1_s1 = s1_name.split()[0] if s1_name else ""
    w1_cand = cand_name.split()[0] if cand_name else ""
    first_word_match = 1.0 if (w1_s1 and w1_cand and w1_s1 == w1_cand) else 0.0
    
    # 2. Address Similarities
    has_addr = bool(s1_addr and cand_addr)
    addr_lev = distance.Levenshtein.normalized_similarity(s1_addr, cand_addr) if has_addr else 0.0
    addr_jw = distance.JaroWinkler.similarity(s1_addr, cand_addr) if has_addr else 0.0
    addr_tsort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0 if has_addr else 0.0
    addr_tset = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0 if has_addr else 0.0
    
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
    
    # Advanced Composite Features
    name_len_diff = abs(len(s1_name) - len(cand_name)) / max(len(s1_name), len(cand_name), 1)
    
    return {
        'name_lev': name_lev, 'name_jw': name_jw, 'name_tsort': name_tsort,
        'name_tset': name_tset, 'name_partial': name_partial, 'name_ratio': name_ratio,
        'first_word_match': first_word_match, 'addr_lev': addr_lev, 'addr_jw': addr_jw,
        'addr_tsort': addr_tsort, 'addr_tset': addr_tset, 'pin_match': pin_match,
        'pin_mismatch': pin_mismatch, 'num_match': num_match, 'num_mismatch': num_mismatch,
        'country_match': country_match, 'country_mismatch': country_mismatch,
        'has_addr': 1.0 if has_addr else 0.0,
        'name_len_diff': name_len_diff,
        'combo_score': 0.5 * name_tset + 0.3 * name_jw + 0.2 * addr_tset
    }

def main():
    dataset_dir = '6ab10eb3b23ba_student_resource/student_resource/dataset'
    sample_size = 5000
    train_dir = os.path.join(dataset_dir, 'train')
    
    s1_df = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t', nrows=sample_size)
    gt_dict = parse_ground_truth(os.path.join(train_dir, 'train_ground_truth.tsv'))
    
    needed_ids = set()
    for sid in s1_df['entity_id']:
        needed_ids.update(gt_dict.get(sid, set()))
        
    targets_df = load_true_match_targets(dataset_dir, needed_ids, bg_samples=25000)

    for df in [s1_df, targets_df]:
        df['business_name_clean'] = df['business_name'].apply(normalize_text)
        df['business_address_clean'] = df['business_address'].apply(normalize_text)

    s1_ids = s1_df['entity_id'].tolist()
    train_ids, val_ids = train_test_split(s1_ids, test_size=0.20, random_state=42)
    
    s1_train_df = s1_df[s1_df['entity_id'].isin(train_ids)].copy()
    s1_val_df = s1_df[s1_df['entity_id'].isin(val_ids)].copy()
    val_gt_dict = {sid: gt_dict.get(sid, set()) for sid in val_ids}

    val_candidates = generate_multi_strategy_candidates(s1_val_df, targets_df, top_k=40)
    train_candidates = generate_multi_strategy_candidates(s1_train_df, targets_df, top_k=40)
    
    s1_train_dict = s1_train_df.set_index('entity_id').to_dict('index')
    val_s1_dict = s1_val_df.set_index('entity_id').to_dict('index')
    target_dict = targets_df.set_index('entity_id').to_dict('index')

    X_train_rows, y_train_list = [], []
    for sid in s1_train_df['entity_id']:
        true_set = gt_dict.get(sid, set())
        cands = train_candidates.get(sid, set())
        s1_row = s1_train_dict[sid]
        for cid in cands:
            if cid in target_dict:
                cand_row = target_dict[cid]
                feats = extract_features(s1_row, cand_row)
                is_match = 1.0 if cid in true_set else 0.0
                X_train_rows.append(feats)
                y_train_list.append(is_match)
                
    X_train = pd.DataFrame(X_train_rows)
    y_train = np.array(y_train_list)

    base_lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=8, num_leaves=63, random_state=42, n_jobs=-1, verbosity=-1)
    clf = CalibratedClassifierCV(estimator=base_lgbm, cv=3, method='isotonic')
    clf.fit(X_train, y_train)

    val_rows = []
    for sid in s1_val_df['entity_id']:
        cands = val_candidates.get(sid, set())
        s1_row = val_s1_dict[sid]
        for cid in cands:
            if cid in target_dict:
                cand_row = target_dict[cid]
                feats = extract_features(s1_row, cand_row)
                feats['source1_entity_id'] = sid
                feats['candidate_entity_id'] = cid
                val_rows.append(feats)

    val_df = pd.DataFrame(val_rows)
    feature_cols = [c for c in val_df.columns if c not in ['source1_entity_id', 'candidate_entity_id']]
    probs = clf.predict_proba(val_df[feature_cols])[:, 1]
    val_df['prob'] = probs

    best_score = -1.0
    best_tau = 0.50

    for tau in np.arange(0.20, 0.95, 0.01):
        preds = defaultdict(set)
        matched_df = val_df[val_df['prob'] >= tau]
        for row in matched_df.itertuples():
            preds[row.source1_entity_id].add(row.candidate_entity_id)
            
        score = evaluate_predictions_macro_f05(val_gt_dict, preds)
        if score > best_score:
            best_score = score
            best_tau = tau

    print(f"\n[SUMMARY RESULTS]")
    print(f"Optimal Threshold tau: {best_tau:.3f}")
    print(f"Validation Macro F_0.5 Score: {best_score:.4f} ({best_score*100:.2f}%)")

    # Error analysis: inspect worst 5 validation entities
    best_preds = defaultdict(set)
    matched_df = val_df[val_df['prob'] >= best_tau]
    for row in matched_df.itertuples():
        best_preds[row.source1_entity_id].add(row.candidate_entity_id)

    errors = []
    for sid, true_set in val_gt_dict.items():
        pred_set = best_preds.get(sid, set())
        score = compute_entity_f05(true_set, pred_set)
        if score < 1.0:
            errors.append((sid, score, true_set, pred_set))

    print(f"\nTotal Error Entities in Val (F0.5 < 1.0): {len(errors)} / {len(val_ids)}")
    print("Sample Error Cases:")
    for sid, sc, tset, pset in errors[:5]:
        s1_r = val_s1_dict[sid]
        print(f"\nS1 ID {sid} (F0.5={sc:.3f}):")
        print(f"  Name: {s1_r['business_name']}")
        print(f"  Addr: {s1_r['business_address']}")
        print(f"  True Matches ({len(tset)}): {tset}")
        print(f"  Pred Matches ({len(pset)}): {pset}")
        
        # Print feature details for true matches vs false positives
        for cid in tset.union(pset):
            if cid in target_dict:
                cr = target_dict[cid]
                is_t = cid in tset
                is_p = cid in pset
                prob_v = val_df[(val_df['source1_entity_id']==sid) & (val_df['candidate_entity_id']==cid)]['prob'].values
                pr = prob_v[0] if len(prob_v) > 0 else 0.0
                print(f"    Cand {cid} [True={is_t}, Pred={is_p}, Prob={pr:.3f}]: Name='{cr['business_name']}', Addr='{cr['business_address']}'")

if __name__ == '__main__':
    main()
