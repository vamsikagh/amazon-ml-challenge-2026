import os
import gc
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Set, Tuple
from .normalization import extract_pin_code

STOP_WORDS = {
    'private', 'limited', 'incorporated', 'corporation', 'company', 
    'the', 'and', 'of', 'in', 'for', 'at', 'on', 'inc', 'ltd', 
    'pvt', 'llp', 'llc', 'co', 'services', 'traders', 'trading', 
    'enterprises', 'group', 'solutions', 'store', 'stores', 'shop', 
    'market', 'center', 'centre', 'india', 'delhi', 'texas', 'city'
}

def extract_meaningful_tokens(text: str) -> List[str]:
    """Extracts non-stopword meaningful tokens (len >= 3)."""
    if not isinstance(text, str):
        return []
    return [t for t in text.split() if len(t) >= 3 and t not in STOP_WORDS]

def generate_token_candidates_low_mem(
    s1_df: pd.DataFrame, 
    source_df: pd.DataFrame,
    max_token_freq: int = 1500,
    max_cands_per_entity: int = 50
) -> Dict[str, Set[str]]:
    """
    Low-Memory Inverted Index Candidate Generator (< 4GB RAM).
    Uses integer ID indexing & high-frequency token filtering to prevent MemoryError.
    """
    # Create fast integer ID mapping for targets to save RAM
    target_ids = source_df['entity_id'].values
    n_targets = len(target_ids)
    
    # 1. Frequency Counter to identify & drop high-frequency generic tokens
    token_freq = Counter()
    for name in source_df['business_name_clean'].fillna(''):
        for t in extract_meaningful_tokens(name):
            token_freq[t] += 1
            
    # Filter out tokens appearing in > max_token_freq entities
    valid_tokens = {t for t, count in token_freq.items() if count <= max_token_freq}
    del token_freq
    gc.collect()

    # 2. Build Memory-Light Inverted Index storing integer indices
    name_index = defaultdict(list)
    prefix_index = defaultdict(list)
    pin_index = defaultdict(list)

    for idx, row in enumerate(source_df.itertuples()):
        name = getattr(row, 'business_name_clean', '')
        addr = getattr(row, 'business_address_clean', '')
        
        # Name Token Index
        tokens = [t for t in extract_meaningful_tokens(name) if t in valid_tokens]
        for token in tokens:
            name_index[token].append(idx)
            
        # Name Prefix Index (if prefix is rare)
        if len(name) >= 4:
            pref = name[:4]
            prefix_index[pref].append(idx)
            
        # Address PIN Code Index
        pin = extract_pin_code(addr)
        if pin:
            pin_index[pin].append(idx)

    candidates = defaultdict(set)

    # 3. Candidate Matching for S1 Entities
    for s1_row in s1_df.itertuples():
        sid = s1_row.entity_id
        name = getattr(s1_row, 'business_name_clean', '')
        addr = getattr(s1_row, 'business_address_clean', '')
        
        s1_cands = set()
        
        # Token Candidates
        tokens = [t for t in extract_meaningful_tokens(name) if t in valid_tokens]
        for token in tokens:
            if token in name_index:
                s1_cands.update(name_index[token][:100]) # Cap per token bucket
                
        # Prefix Candidates
        if len(name) >= 4:
            pref = name[:4]
            if pref in prefix_index and len(prefix_index[pref]) <= max_token_freq:
                s1_cands.update(prefix_index[pref][:100])
                
        # PIN Candidates
        pin = extract_pin_code(addr)
        if pin and pin in pin_index and len(pin_index[pin]) <= max_token_freq:
            s1_cands.update(pin_index[pin][:100])
            
        # Cap candidates per S1 entity and convert integer indices to string entity_ids
        if s1_cands:
            s1_cand_list = list(s1_cands)[:max_cands_per_entity]
            candidates[sid] = {target_ids[i] for i in s1_cand_list}

    # Clean up intermediate dicts
    del name_index, prefix_index, pin_index
    gc.collect()
    
    return candidates

def generate_tfidf_candidates_low_mem(
    s1_df: pd.DataFrame, 
    target_df: pd.DataFrame, 
    top_k: int = 30,
    batch_size: int = 200000
) -> Dict[str, Set[str]]:
    """
    Computes top-k candidates per S1 entity using 3-gram character TF-IDF cosine similarity in batches.
    """
    s1_texts = (s1_df['business_name_clean'].fillna('') + ' ' + s1_df['business_address_clean'].fillna('')).tolist()
    target_texts = (target_df['business_name_clean'].fillna('') + ' ' + target_df['business_address_clean'].fillna('')).tolist()
    target_ids = target_df['entity_id'].values
    
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=5, max_features=100000)
    vectorizer.fit(target_texts[:300000]) # Fit sample for high speed & low memory
    
    target_vecs = vectorizer.transform(target_texts)
    
    nn = NearestNeighbors(n_neighbors=min(top_k, target_vecs.shape[0]), metric='cosine', algorithm='brute')
    nn.fit(target_vecs)
    
    candidates = defaultdict(set)
    s1_ids = s1_df['entity_id'].tolist()
    
    # Process S1 entities in batches to save memory
    for start_idx in range(0, len(s1_texts), batch_size):
        end_idx = min(start_idx + batch_size, len(s1_texts))
        batch_vecs = vectorizer.transform(s1_texts[start_idx:end_idx])
        distances, indices = nn.kneighbors(batch_vecs)
        
        for idx_offset, neighbor_indices in enumerate(indices):
            sid = s1_ids[start_idx + idx_offset]
            candidates[sid].update([target_ids[n_idx] for n_idx in neighbor_indices])
            
    return candidates

def generate_ultra_high_recall_blocking_candidates(
    s1_df: pd.DataFrame, 
    target_df: pd.DataFrame, 
    top_k_tfidf: int = 30,
    max_candidates_cap: int = 60
) -> Dict[str, Set[str]]:
    """
    Low-Memory Union of Inverted Indexing + Batch Character TF-IDF Cosine Nearest Neighbors.
    Optimized to run under 4GB RAM with zero MemoryError.
    """
    cands_token = generate_token_candidates_low_mem(s1_df, target_df, max_cands_per_entity=max_candidates_cap)
    cands_tfidf = generate_tfidf_candidates_low_mem(s1_df, target_df, top_k=top_k_tfidf)

    union_cands = {}
    s1_ids = s1_df['entity_id'].tolist()
    for s1_id in s1_ids:
        u_set = cands_token.get(s1_id, set()).union(cands_tfidf.get(s1_id, set()))
        if len(u_set) > max_candidates_cap:
            union_cands[s1_id] = set(list(u_set)[:max_candidates_cap])
        else:
            union_cands[s1_id] = u_set

    return union_cands

def save_candidate_pairs_tsv(candidates: Dict[str, Set[str]], all_s1_ids: List[str], output_path: str):
    """Saves candidate_pairs.tsv ensuring exactly 1 row per S1 entity."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for s1_id in all_s1_ids:
        cand_list = sorted(list(candidates.get(s1_id, set())))
        cand_str = ",".join(cand_list)
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_str})
        
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
