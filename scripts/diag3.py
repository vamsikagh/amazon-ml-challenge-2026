"""Lean, fast recall diagnostic (400k pool). Answers: does transliteration +
dedicated address blocking + union beat the current name-centric combined blocking?"""
import os, sys, time, re, csv, unicodedata
import numpy as np, pandas as pd
import nmslib
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
DATA = os.environ.get("DATASET_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"))
T0=time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}",flush=True)
BG=150000; NS1=3000

LEGAL={r'\bpvt\.?\b':'private',r'\bltd\.?\b':'limited',r'\binc\.?\b':'incorporated',r'\bcorp\.?\b':'corporation',r'\bllp\.?\b':'llp',r'\bllc\.?\b':'llc'}
ADDR={r'\brd\.?\b':'road',r'\bst\.?\b':'street',r'\bave\.?\b':'avenue',r'\bblvd\.?\b':'boulevard'}
def clean(t):
    for p,r in {**LEGAL,**ADDR}.items(): t=re.sub(p,r,t)
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9\s]',' ',t)).strip()
def n_ascii(t):
    if not isinstance(t,str) or not t.strip(): return ""
    return clean(unicodedata.normalize('NFKD',t).encode('ASCII','ignore').decode().lower())
def n_tl(t):
    if not isinstance(t,str) or not t.strip(): return ""
    return clean(unidecode(t).lower())

gt={}
with open(f"{DATA}/train/train_ground_truth.tsv") as f:
    r=csv.reader(f,delimiter='\t'); next(r)
    for row in r:
        m=[x for x in (row[1] if len(row)>1 else "").split(',') if x]
        if m: gt[row[0]]=set(m)
        if len(gt)>=NS1: break
need=set()
for v in gt.values(): need|=v
def load_src(path,bg):
    matched=[c[c['entity_id'].isin(need)] for c in pd.read_csv(path,sep='\t',chunksize=500000)]
    return pd.concat(matched+[pd.read_csv(path,sep='\t',nrows=bg)],ignore_index=True)
tg=pd.concat([load_src(f"{DATA}/train/train_source2.tsv",BG//2),load_src(f"{DATA}/train/train_source3.tsv",BG//2)],ignore_index=True).drop_duplicates('entity_id')
s1=pd.concat([c[c['entity_id'].isin(gt)] for c in pd.read_csv(f"{DATA}/train/train_source1.tsv",sep='\t',chunksize=500000)],ignore_index=True)
ctry=dict(zip(s1['entity_id'],s1['country']))
log(f"pool {len(tg)} targets, {len(s1)} s1")

def cands(s_txt,t_txt,tid,k):
    vec=TfidfVectorizer(analyzer='char',ngram_range=(3,3),min_df=5,max_features=100000)
    vec.fit(t_txt[:300000] if len(t_txt)>300000 else t_txt)
    idx=nmslib.init(method='hnsw',space='cosinesimil_sparse',data_type=nmslib.DataType.SPARSE_VECTOR)
    idx.addDataPointBatch(vec.transform(t_txt)); idx.createIndex({'M':24,'efConstruction':150,'post':1},print_progress=False)
    idx.setQueryTimeParams({'efSearch':max(300,k)})
    res=idx.knnQueryBatch(vec.transform(s_txt),k=k,num_threads=0)
    return {s1['entity_id'].iloc[i]:set(tid[j] for j in res[i][0]) for i in range(len(s1))}
def rec(c,cc=None):
    tp=tot=0
    for sid,tr in gt.items():
        if cc and ctry.get(sid)!=cc: continue
        tp+=len(tr&c.get(sid,set())); tot+=len(tr)
    return tp/tot if tot else 0
def report(name,c):
    log(f"  {name:26s} all={rec(c)*100:.1f}% US={rec(c,'US')*100:.0f}% India={rec(c,'India')*100:.0f}% France={rec(c,'France')*100:.0f}% avg={np.mean([len(c[s]) for s in c]):.0f}")

tid=tg['entity_id'].values
# baseline: current ASCII combined k30
an=tg['business_name'].apply(n_ascii).values; aa=tg['business_address'].apply(n_ascii).values
sn=s1['business_name'].apply(n_ascii).values; sa=s1['business_address'].apply(n_ascii).values
report("ASCII combined k30", cands([f"{n} {a}" for n,a in zip(sn,sa)],[f"{n} {a}" for n,a in zip(an,aa)],tid,30))
log("--- transliterate ---")
tn=tg['business_name'].apply(n_tl).values; ta=tg['business_address'].apply(n_tl).values
qn=s1['business_name'].apply(n_tl).values; qa=s1['business_address'].apply(n_tl).values
c_comb=cands([f"{n} {a}" for n,a in zip(qn,qa)],[f"{n} {a}" for n,a in zip(tn,ta)],tid,60); report("TL combined k60",c_comb)
c_addr=cands(list(qa),list(ta),tid,50); report("TL addr k50",c_addr)
c_name=cands(list(qn),list(tn),tid,50); report("TL name k50",c_name)
u={sid:c_comb.get(sid,set())|c_addr.get(sid,set())|c_name.get(sid,set()) for sid in gt}
report("TL UNION(cmb+addr+name)",u)
log("DONE")
