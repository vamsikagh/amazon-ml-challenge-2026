"""Train the calibrated classifier, optimise the F_0.5 threshold on a validation
split, then apply it (multi-match) to the checkpointed TEST candidate_pairs.tsv
to regenerate matching_results.tsv. Reuses the blocking already done for test."""
import os, sys, time, gc, argparse, pickle, csv, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = BASE
sys.path.insert(0, os.path.join(REPO, "code/business_entity_resolution"))
DATA = os.environ.get("DATASET_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"))
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("OUTPUT_DIR", os.path.join(SCR, "output"))

from src.normalization import normalize_text
from src.blocking import generate_ultra_high_recall_blocking_candidates
from src.features import extract_pair_features
from src.evaluation import evaluate_predictions_macro_f05, verify_evaluator_against_worked_example
from src.classifier import EntityResolutionClassifier

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

def parse_gt(path):
    d = {}
    with open(path) as f:
        r = csv.reader(f, delimiter='\t'); next(r)
        for row in r:
            m = set(x for x in (row[1] if len(row) > 1 else "").split(',') if x)
            d[row[0]] = m
    return d

def load_true_match_targets(needed_ids, bg_samples):
    tr = os.path.join(DATA, 'train')
    parts = []
    for src in ('train_source2.tsv', 'train_source3.tsv'):
        for chunk in pd.read_csv(os.path.join(tr, src), sep='\t', chunksize=500000):
            m = chunk[chunk['entity_id'].isin(needed_ids)]
            if len(m): parts.append(m)
    parts.append(pd.read_csv(os.path.join(tr, 'train_source2.tsv'), sep='\t', nrows=bg_samples))
    parts.append(pd.read_csv(os.path.join(tr, 'train_source3.tsv'), sep='\t', nrows=bg_samples))
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=['entity_id'])

def train_classifier(sample_size):
    assert verify_evaluator_against_worked_example()
    log(f"TRAIN: loading {sample_size} train S1 entities")
    s1 = pd.read_csv(os.path.join(DATA, 'train/train_source1.tsv'), sep='\t', nrows=sample_size)
    gt = parse_gt(os.path.join(DATA, 'train/train_ground_truth.tsv'))
    needed = set()
    for sid in s1['entity_id']: needed |= gt.get(sid, set())
    targets = load_true_match_targets(needed, bg_samples=sample_size * 5)
    log(f"TRAIN: target pool {len(targets)}")
    from sklearn.model_selection import train_test_split
    tr_ids, val_ids = train_test_split(s1['entity_id'].tolist(), test_size=0.2, random_state=42)
    s1_tr = s1[s1['entity_id'].isin(tr_ids)].copy()
    s1_val = s1[s1['entity_id'].isin(val_ids)].copy()
    val_gt = {i: gt.get(i, set()) for i in val_ids}
    for df in (s1_tr, s1_val, targets):
        df['business_name_clean'] = df['business_name'].apply(normalize_text)
        df['business_address_clean'] = df['business_address'].apply(normalize_text)
    log("TRAIN: blocking train+val")
    tr_cand = generate_ultra_high_recall_blocking_candidates(s1_tr, targets, top_k_tfidf=30, max_candidates_cap=60)
    val_cand = generate_ultra_high_recall_blocking_candidates(s1_val, targets, top_k_tfidf=30, max_candidates_cap=60)
    tname = dict(zip(targets['entity_id'], targets['business_name_clean']))
    taddr = dict(zip(targets['entity_id'], targets['business_address_clean']))
    tctry = dict(zip(targets['entity_id'], targets['country'].fillna('').astype(str)))
    def trow(i): return {'business_name_clean': tname[i], 'business_address_clean': taddr[i], 'country': tctry[i]}
    def srow(r): return {'business_name_clean': r.business_name_clean, 'business_address_clean': r.business_address_clean, 'country': str(r.country)}
    log("TRAIN: extracting features")
    feats, labels = [], []
    s1tr_idx = {r.entity_id: srow(r) for r in s1_tr.itertuples()}
    for sid in s1_tr['entity_id']:
        tset = gt.get(sid, set()); sr = s1tr_idx[sid]
        for cid in tr_cand.get(sid, set()):
            if cid in tname:
                feats.append(extract_pair_features(sr, trow(cid)))
                labels.append(1.0 if cid in tset else 0.0)
    X = pd.DataFrame(feats); y = np.array(labels)
    log(f"TRAIN: {len(X)} pairs ({int(y.sum())} pos). Fitting classifier...")
    clf = EntityResolutionClassifier(n_estimators=300)
    clf.fit(X, y)
    feat_cols = list(X.columns)
    # optimize threshold on val
    s1val_idx = {r.entity_id: srow(r) for r in s1_val.itertuples()}
    vrows = []
    for sid in s1_val['entity_id']:
        sr = s1val_idx[sid]
        for cid in val_cand.get(sid, set()):
            if cid in tname:
                f = extract_pair_features(sr, trow(cid)); f['source1_entity_id'] = sid; f['candidate_entity_id'] = cid
                vrows.append(f)
    vdf = pd.DataFrame(vrows)
    best_f05, best_tau = clf.optimize_threshold_f05(vdf, val_gt)
    log(f"TRAIN: val macro F_0.5={best_f05:.4f} at tau={best_tau:.3f}")
    return clf, best_tau, feat_cols

def build_test_lookups():
    log("INFER: loading test data")
    s1 = pd.read_csv(os.path.join(DATA, 'test/test_source1.tsv'), sep='\t')
    s2 = pd.read_csv(os.path.join(DATA, 'test/test_source2.tsv'), sep='\t')
    s3 = pd.read_csv(os.path.join(DATA, 'test/test_source3.tsv'), sep='\t')
    tg = pd.concat([s2, s3], ignore_index=True); del s2, s3; gc.collect()
    log("INFER: normalizing")
    for df in (s1, tg):
        df['business_name_clean'] = df['business_name'].apply(normalize_text)
        df['business_address_clean'] = df['business_address'].apply(normalize_text)
    tname = dict(zip(tg['entity_id'], tg['business_name_clean']))
    taddr = dict(zip(tg['entity_id'], tg['business_address_clean']))
    tctry = dict(zip(tg['entity_id'], tg['country'].fillna('').astype(str)))
    s1n = dict(zip(s1['entity_id'], s1['business_name_clean']))
    s1a = dict(zip(s1['entity_id'], s1['business_address_clean']))
    s1c = dict(zip(s1['entity_id'], s1['country'].fillna('').astype(str)))
    del tg, s1; gc.collect()
    return (tname, taddr, tctry), (s1n, s1a, s1c)

def infer(clf, tau, feat_cols, tlk, slk, batch=400000, max_rows=None, res_name='matching_results.tsv'):
    tname, taddr, tctry = tlk
    s1n, s1a, s1c = slk
    cand_path = os.path.join(OUT, 'candidate_pairs.tsv')
    res_path = os.path.join(OUT, res_name)
    log(f"INFER: streaming candidates, tau={tau:.3f}")
    fout = open(res_path, 'w', newline='')
    w = csv.writer(fout, delimiter='\t'); w.writerow(['source1_entity_id', 'matched_entity_ids'])
    buf_feats = []; buf_idx = []   # feature dicts and (row_pos, cand_id)
    pending = {}                   # s1_id -> list of (cand_id) awaiting; we resolve in order
    order = []                     # s1_ids in file order
    results = {}                   # s1_id -> set of matched
    n_pairs = 0; n_rows = 0

    def flush():
        nonlocal buf_feats, buf_idx
        if not buf_feats: return
        X = pd.DataFrame(buf_feats)
        for c in feat_cols:
            if c not in X.columns: X[c] = 0.0
        probs = clf.predict_proba(X[feat_cols])
        for (sid, cid), p in zip(buf_idx, probs):
            if p >= tau:
                results[sid].add(cid)
        buf_feats = []; buf_idx = []

    with open(cand_path) as f:
        r = csv.reader(f, delimiter='\t'); next(r)
        for row in r:
            if max_rows and n_rows >= max_rows: break
            sid = row[0]; order.append(sid); results[sid] = set()
            n_rows += 1
            cands = row[1].split(',') if len(row) > 1 and row[1] else []
            sr = {'business_name_clean': s1n.get(sid, ''), 'business_address_clean': s1a.get(sid, ''), 'country': s1c.get(sid, '')}
            for cid in cands:
                if cid in tname:
                    buf_feats.append(extract_pair_features(sr, {'business_name_clean': tname[cid], 'business_address_clean': taddr[cid], 'country': tctry[cid]}))
                    buf_idx.append((sid, cid)); n_pairs += 1
            if len(buf_feats) >= batch:
                flush()
                if n_rows % 200000 < 1: log(f"INFER: {n_rows} entities, {n_pairs} pairs scored")
    flush()
    log(f"INFER: writing {len(order)} rows")
    for sid in order:
        m = results.get(sid, set())
        w.writerow([sid, ",".join(sorted(m))])
    fout.close()
    matched = sum(1 for s in order if results.get(s))
    log(f"INFER: done. entities={len(order)} matched={matched} ({100*matched/len(order):.1f}%) pairs={n_pairs}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-size', type=int, default=15000)
    ap.add_argument('--batch', type=int, default=400000)
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--res-name', type=str, default='matching_results.tsv')
    args = ap.parse_args()
    clf, tau, feat_cols = train_classifier(args.sample_size)
    with open(os.path.join(SCR, 'clf.pkl'), 'wb') as f:
        pickle.dump({'clf': clf, 'tau': tau, 'feat_cols': feat_cols}, f)
    log("saved clf.pkl")
    tlk, slk = build_test_lookups()
    infer(clf, tau, feat_cols, tlk, slk, batch=args.batch, max_rows=args.max_rows, res_name=args.res_name)
    log("ALL DONE")
