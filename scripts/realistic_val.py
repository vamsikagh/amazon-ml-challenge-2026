"""Realistic-scale validation: train classifier + evaluate official macro F_0.5
on a large-distractor pool (mirrors the 10M test far better than the tiny pool
that lied to us). Reports recall, best threshold, and F_0.5 for the NEW pipeline."""
import os, sys, time, gc, csv, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = BASE
sys.path.insert(0, os.path.join(REPO,"code/business_entity_resolution"))
DATA = os.environ.get("DATASET_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"))
from src.normalization import normalize_text
from src.blocking import generate_ultra_high_recall_blocking_candidates
from src.features import extract_pair_features
from src.evaluation import evaluate_predictions_macro_f05
from src.classifier import EntityResolutionClassifier
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}",flush=True)

N_TRAIN=5000; N_VAL=3000; BG=2000000  # 2M background distractors

gt={}
with open(f"{DATA}/train/train_ground_truth.tsv") as f:
    r=csv.reader(f,delimiter='\t'); next(r)
    for row in r:
        m=set(x for x in (row[1] if len(row)>1 else "").split(',') if x)
        gt[row[0]]=m

# take first N_TRAIN+N_VAL S1 entities that HAVE matches
ids=[k for k in gt if gt[k]][:N_TRAIN+N_VAL]
tr_ids=set(ids[:N_TRAIN]); val_ids=set(ids[N_TRAIN:])
need=set()
for i in ids: need|=gt[i]
log(f"train={len(tr_ids)} val={len(val_ids)} need={len(need)} match ids")

def load_src(path,bg):
    matched=[c[c['entity_id'].isin(need)] for c in pd.read_csv(path,sep='\t',chunksize=1000000)]
    return pd.concat(matched+[pd.read_csv(path,sep='\t',nrows=bg)],ignore_index=True)
log("loading pool (scanning full source files for true matches + background)...")
tg=pd.concat([load_src(f"{DATA}/train/train_source2.tsv",BG//2),load_src(f"{DATA}/train/train_source3.tsv",BG//2)],ignore_index=True).drop_duplicates('entity_id')
allids=tr_ids|val_ids
s1=pd.concat([c[c['entity_id'].isin(allids)] for c in pd.read_csv(f"{DATA}/train/train_source1.tsv",sep='\t',chunksize=1000000)],ignore_index=True)
log(f"pool: {len(tg)} targets, {len(s1)} s1")

log("normalizing...")
for df in (s1,tg):
    df['business_name_clean']=df['business_name'].apply(normalize_text)
    df['business_address_clean']=df['business_address'].apply(normalize_text)
s1_tr=s1[s1['entity_id'].isin(tr_ids)].copy(); s1_val=s1[s1['entity_id'].isin(val_ids)].copy()

log("blocking (train) on 2M pool...")
tr_cand=generate_ultra_high_recall_blocking_candidates(s1_tr,tg,top_k_tfidf=30,max_candidates_cap=80)
log("blocking (val) on 2M pool...")
val_cand=generate_ultra_high_recall_blocking_candidates(s1_val,tg,top_k_tfidf=30,max_candidates_cap=80)

# recall gate on val
tp=tot=0
for sid in val_ids:
    tp+=len(gt[sid]&val_cand.get(sid,set())); tot+=len(gt[sid])
log(f"VAL BLOCKING RECALL = {tp/tot*100:.2f}% ({tp}/{tot})")

tname=dict(zip(tg['entity_id'],tg['business_name_clean'])); taddr=dict(zip(tg['entity_id'],tg['business_address_clean'])); tctry=dict(zip(tg['entity_id'],tg['country'].fillna('').astype(str)))
def trow(i): return {'business_name_clean':tname[i],'business_address_clean':taddr[i],'country':tctry[i]}
def srow(r): return {'business_name_clean':r.business_name_clean,'business_address_clean':r.business_address_clean,'country':str(r.country)}

log("extracting train features (hard negatives from 2M pool)...")
feats=[]; labels=[]
tri={r.entity_id:srow(r) for r in s1_tr.itertuples()}
for sid in tr_ids:
    tset=gt[sid]; sr=tri[sid]
    for cid in tr_cand.get(sid,set()):
        if cid in tname:
            feats.append(extract_pair_features(sr,trow(cid))); labels.append(1.0 if cid in tset else 0.0)
X=pd.DataFrame(feats); y=np.array(labels)
log(f"train pairs={len(X)} pos={int(y.sum())} ({100*y.mean():.1f}%). fitting...")
clf=EntityResolutionClassifier(n_estimators=300); clf.fit(X,y)
feat_cols=list(X.columns)

log("scoring val...")
vi={r.entity_id:srow(r) for r in s1_val.itertuples()}
vrows=[]
for sid in val_ids:
    sr=vi[sid]
    for cid in val_cand.get(sid,set()):
        if cid in tname:
            f=extract_pair_features(sr,trow(cid)); f['s']=sid; f['c']=cid; vrows.append(f)
vdf=pd.DataFrame(vrows)
probs=clf.predict_proba(vdf[feat_cols]); vdf['prob']=probs
val_gt={i:gt[i] for i in val_ids}
best=(0,0)
for tau in np.arange(0.30,0.95,0.02):
    preds={}
    sel=vdf[vdf['prob']>=tau]
    for row in sel.itertuples():
        preds.setdefault(row.s,set()).add(row.c)
    f05=evaluate_predictions_macro_f05(val_gt,preds)
    if f05>best[0]: best=(f05,tau)
log(f"*** REALISTIC VAL: best macro F_0.5 = {best[0]:.4f} at tau={best[1]:.2f} ***")
# also report F_0.5 at a few taus
for tau in [0.4,0.5,0.6,0.7,0.8]:
    preds={}
    for row in vdf[vdf['prob']>=tau].itertuples(): preds.setdefault(row.s,set()).add(row.c)
    log(f"   tau={tau}: F_0.5={evaluate_predictions_macro_f05(val_gt,preds):.4f}")
log("DONE")
