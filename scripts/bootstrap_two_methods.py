#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def load(paths, label):
    rows=[]
    for raw in paths:
        path=Path(raw)
        with path.open() as f:
            for line in f:
                r=json.loads(line)
                rows.append({"id":r.get("id"),"benchmark":r.get("benchmark"),"seed":int(r.get("seed",0)),"method":label,"correct":int(bool(r.get("correct"))),"confidence":float(r.get("confidence",0.0))})
    return pd.DataFrame(rows)

def ece(conf,corr,bins=10):
    conf=np.asarray(conf,dtype=float); corr=np.asarray(corr,dtype=float); n=len(conf); val=0.0
    for b in range(bins):
        lo=b/bins; hi=(b+1)/bins
        m=(conf>=lo)&((conf<=hi) if b==bins-1 else (conf<hi))
        if m.any(): val += (m.sum()/n)*abs(corr[m].mean()-conf[m].mean())
    return float(val)

def aurc(conf,corr):
    conf=np.asarray(conf,dtype=float); corr=np.asarray(corr,dtype=float); order=np.argsort(-conf); c=corr[order]
    return float((1.0-np.cumsum(c)/np.arange(1,len(c)+1)).mean())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--a',nargs='+',required=True); ap.add_argument('--b',nargs='+',required=True)
    ap.add_argument('--a_label',default='A'); ap.add_argument('--b_label',default='B')
    ap.add_argument('--out',required=True); ap.add_argument('--boot',type=int,default=2000)
    args=ap.parse_args()
    a=load(args.a,args.a_label); b=load(args.b,args.b_label)
    merged=a.merge(b,on=['seed','benchmark','id'],suffixes=('_a','_b'))
    if merged.empty: raise SystemExit('No paired rows after merge. Check id/seed/benchmark alignment.')
    rng=np.random.default_rng(13); n=len(merged)
    def point(s):
        return {f'FA ({args.a_label} - {args.b_label})':s.correct_a.mean()-s.correct_b.mean(),
                f'ECE10 ({args.a_label} - {args.b_label})':ece(s.confidence_a,s.correct_a)-ece(s.confidence_b,s.correct_b),
                f'AURC ({args.a_label} - {args.b_label})':aurc(s.confidence_a,s.correct_a)-aurc(s.confidence_b,s.correct_b)}
    pe=point(merged); boots={k:[] for k in pe}
    for _ in range(args.boot):
        sample=merged.iloc[rng.integers(0,n,n)]; q=point(sample)
        for k,v in q.items(): boots[k].append(v)
    rows=[]
    for k,vals in boots.items():
        lo,med,hi=np.percentile(vals,[2.5,50,97.5]); rows.append({'metric':k,'point':pe[k],'boot_p2_5':lo,'boot_median':med,'boot_p97_5':hi,'n_paired':n})
    out=pd.DataFrame(rows); Path(args.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.out,index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
