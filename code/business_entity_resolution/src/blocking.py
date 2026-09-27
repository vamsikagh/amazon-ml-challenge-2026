import os
import gc
import math
import pickle
import tempfile
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Set, Tuple
from .normalization import extract_pin_code

try:
    import nmslib
    _NMSLIB_AVAILABLE = True
except ImportError:
    _NMSLIB_AVAILABLE = False

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

# Common address words that are NOT discriminative (dropped from address blocking).
ADDR_STOP = {
    'road', 'street', 'avenue', 'boulevard', 'lane', 'drive', 'nagar', 'colony',
    'near', 'opposite', 'behind', 'main', 'cross', 'block', 'sector', 'phase',
    'floor', 'building', 'apartment', 'number', 'po', 'box', 'null', 'na',
    'north', 'south', 'east', 'west', 'new', 'old', 'the', 'and',
}

def extract_addr_tokens(text: str) -> List[str]:
    """Discriminative address tokens: street numbers (>=2 digits) and locality/
    street names (len>=4, non-stopword). Street numbers and unusual locality names
    are strong join keys shared across a business's records even when the NAME is
    transliterated, a website, or heavily typo'd."""
    if not isinstance(text, str):
        return []
    out = []
    for t in text.split():
        if t.isdigit():
            if len(t) >= 2:
                out.append(t)
        elif len(t) >= 4 and t not in ADDR_STOP and t not in STOP_WORDS:
            out.append(t)
    return out

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
    addr_freq = Counter()
    for name, addr in zip(source_df['business_name_clean'].fillna(''),
                          source_df['business_address_clean'].fillna('')):
        for t in extract_meaningful_tokens(name):
            token_freq[t] += 1
        for t in extract_addr_tokens(addr):
            addr_freq[t] += 1

    # Filter out tokens appearing in > max_token_freq entities
    valid_tokens = {t for t, count in token_freq.items() if count <= max_token_freq}
    valid_addr = {t for t, count in addr_freq.items() if count <= max_token_freq}
    del token_freq, addr_freq
    gc.collect()

    # 2. Build Memory-Light Inverted Index storing integer indices
    name_index = defaultdict(list)
    prefix_index = defaultdict(list)
    pin_index = defaultdict(list)
    addr_index = defaultdict(list)

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

        # Address Token Index (street numbers + discriminative locality tokens)
        for t in extract_addr_tokens(addr):
            if t in valid_addr:
                addr_index[t].append(idx)

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
                s1_cands.update(name_index[token][:150]) # Cap per token bucket

        # Prefix Candidates
        if len(name) >= 4:
            pref = name[:4]
            if pref in prefix_index and len(prefix_index[pref]) <= max_token_freq:
                s1_cands.update(prefix_index[pref][:150])

        # PIN Candidates
        pin = extract_pin_code(addr)
        if pin and pin in pin_index and len(pin_index[pin]) <= max_token_freq:
            s1_cands.update(pin_index[pin][:150])

        # Address Token Candidates — pairs sharing a street number + a locality
        # token are strong matches; require the address signal via intersection of
        # buckets so we don't flood on a single common token.
        addr_toks = [t for t in extract_addr_tokens(addr) if t in valid_addr and t in addr_index]
        for t in addr_toks:
            s1_cands.update(addr_index[t][:150])

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
    n_shards: int = None,
    shard_target_size: int = 500000,
    transform_chunk: int = 200000,
    query_chunk: int = 200000,
    hnsw_M: int = 24,
    hnsw_ef_construction: int = 150,
    hnsw_ef_query: int = 300,
    hnsw_post: int = 1,
    index_threads: int = None,
    query_threads: int = None,
) -> Dict[str, Set[str]]:
    """
    Approximate top-k candidates per S1 entity via 3-gram character TF-IDF cosine
    similarity, scaled to ~10M targets with a *sharded* sparse HNSW index (nmslib).
    """
    if index_threads is None:
        index_threads = max(1, (os.cpu_count() or 4) - 1)
    if query_threads is None:
        query_threads = max(1, (os.cpu_count() or 4) - 1)
    if not _NMSLIB_AVAILABLE:
        raise ImportError("nmslib is required for approximate blocking; add it to requirements.txt")

    target_ids = target_df['entity_id'].values
    n_t = len(target_ids)
    k = min(top_k, n_t)
    n_q = len(s1_df)

    if n_shards is None:
        n_shards = max(1, math.ceil(n_t / shard_target_size))

    # Fit the vectorizer once (vocab/IDF on a sample of targets, unchanged).
    target_name = target_df['business_name_clean'].fillna('').values
    target_addr = target_df['business_address_clean'].fillna('').values
    fit_sample = [f"{n} {a}" for n, a in zip(target_name[:300000], target_addr[:300000])]
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=5, max_features=100000)
    vectorizer.fit(fit_sample)
    del fit_sample
    gc.collect()

    # Transform all S1 queries once and reuse across shards (~1.4GB sparse).
    s1_ids = s1_df['entity_id'].tolist()
    s1_texts = (s1_df['business_name_clean'].fillna('') + ' ' + s1_df['business_address_clean'].fillna('')).tolist()
    Xq = vectorizer.transform(s1_texts)
    del s1_texts
    gc.collect()

    # Running global top-k per S1 entity (by cosine distance; smaller = closer).
    best_dist = np.full((n_q, k), np.inf, dtype=np.float32)
    best_gid = np.full((n_q, k), -1, dtype=np.int64)

    shard_bounds = list(range(0, n_t, math.ceil(n_t / n_shards)))
    print(f"    [tfidf] {n_shards} shard(s) over {n_t} targets, {n_q} queries", flush=True)
    for si, s0 in enumerate(shard_bounds):
        s_end = min(s0 + math.ceil(n_t / n_shards), n_t)
        print(f"    [tfidf] shard {si+1}/{len(shard_bounds)}: building index on targets {s0}:{s_end}", flush=True)
        index = nmslib.init(method='hnsw', space='cosinesimil_sparse',
                            data_type=nmslib.DataType.SPARSE_VECTOR)
        # add this shard's targets in chunks (local ids 0..shard_len-1)
        for start in range(s0, s_end, transform_chunk):
            end = min(start + transform_chunk, s_end)
            texts = [f"{n} {a}" for n, a in zip(target_name[start:end], target_addr[start:end])]
            chunk = vectorizer.transform(texts)
            index.addDataPointBatch(chunk, ids=np.arange(start - s0, end - s0))
            del texts, chunk
            gc.collect()
        index.createIndex({'M': hnsw_M, 'efConstruction': hnsw_ef_construction, 'post': hnsw_post,
                           'indexThreadQty': index_threads}, print_progress=False)
        index.setQueryTimeParams({'efSearch': max(hnsw_ef_query, k)})
        print(f"    [tfidf] shard {si+1}/{len(shard_bounds)}: index built, querying all S1...", flush=True)

        # query all S1 against this shard, merge into the running global top-k
        for qs in range(0, n_q, query_chunk):
            qe = min(qs + query_chunk, n_q)
            res = index.knnQueryBatch(Xq[qs:qe], k=k, num_threads=query_threads)
            B = qe - qs
            new_dist = np.full((B, k), np.inf, dtype=np.float32)
            new_gid = np.full((B, k), -1, dtype=np.int64)
            for i, (labs, dists) in enumerate(res):
                m = len(labs)
                if m:
                    new_dist[i, :m] = dists
                    new_gid[i, :m] = np.asarray(labs, dtype=np.int64) + s0  # local -> global
            cmb_d = np.concatenate([best_dist[qs:qe], new_dist], axis=1)
            cmb_g = np.concatenate([best_gid[qs:qe], new_gid], axis=1)
            order = np.argsort(cmb_d, axis=1, kind='stable')[:, :k]
            rows = np.arange(B)[:, None]
            best_dist[qs:qe] = cmb_d[rows, order]
            best_gid[qs:qe] = cmb_g[rows, order]
            del res, new_dist, new_gid, cmb_d, cmb_g, order
        del index
        gc.collect()
        print(f"    [tfidf] shard {si+1}/{len(shard_bounds)}: done", flush=True)

    candidates = defaultdict(set)
    for i in range(n_q):
        candidates[s1_ids[i]].update(target_ids[g] for g in best_gid[i] if g >= 0)
    return candidates

def generate_ultra_high_recall_blocking_candidates(
    s1_df: pd.DataFrame, 
    target_df: pd.DataFrame, 
    top_k_tfidf: int = 30,
    max_candidates_cap: int = 60
) -> Dict[str, Set[str]]:
    """
    Union of Inverted-Index blocking + sharded sparse TF-IDF HNSW neighbours.

    Memory-bounded for ~10M targets on a 16GB machine: the token candidates are
    spilled to disk while the (larger) TF-IDF index is built, so the two big
    candidate maps are never resident at the same time as the HNSW index.
    """
    s1_ids = s1_df['entity_id'].tolist()

    # 1. Token/prefix/PIN blocking, then spill to disk and free RAM.
    cands_token = generate_token_candidates_low_mem(s1_df, target_df, max_cands_per_entity=max_candidates_cap)
    spill = os.path.join(tempfile.gettempdir(), f"cands_token_{os.getpid()}.pkl")
    with open(spill, 'wb') as f:
        pickle.dump(dict(cands_token), f, protocol=pickle.HIGHEST_PROTOCOL)
    del cands_token
    gc.collect()

    # 2. Sharded TF-IDF HNSW blocking (peak memory bounded by one shard's index).
    cands_tfidf = generate_tfidf_candidates_low_mem(s1_df, target_df, top_k=top_k_tfidf)

    # 3. Reload token candidates and union, capping per entity.
    with open(spill, 'rb') as f:
        cands_token = pickle.load(f)
    os.remove(spill)

    union_cands = {}
    for s1_id in s1_ids:
        u_set = cands_token.get(s1_id, set()) | cands_tfidf.get(s1_id, set())
        if len(u_set) > max_candidates_cap:
            union_cands[s1_id] = set(sorted(u_set)[:max_candidates_cap])
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
