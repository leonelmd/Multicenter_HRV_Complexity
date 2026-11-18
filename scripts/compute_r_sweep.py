#!/usr/bin/env python3
# Compute the entropy-tolerance sweep feeding Appendix 6.
#
# Two tolerance axes, crossed with embedding dimension m = 1, 2, 3:
#   mode="k"    r = k x SD(subject)   k = 0.05 .. 0.50  (rule universal, value per-patient)
#   mode="abs"  r = fixed absolute ms r = 4 .. 60 ms    (universal in both senses)
#
# Distances are computed ONCE per (subject, scale, shift) and thresholded at every
# tolerance via searchsorted+bincount, so the sweep costs little more than one rcMSE
# run. Split-half halves (h1, h2) are emitted at every tolerance for the reliability
# panel.
#
# Nagoya: full 18:00-22:00 window (the pre-specified analysis window), no beat cap
# (KD-tree counts since 2026-11-18; previously 16:00-20:00 capped at 2000 beats).
# All cohorts use nAUC over scales 1-5 here.
#
# Output: data/r_sweep.csv

import numpy as np, pandas as pd, glob, os
from datetime import datetime

def counts_multi(x, m, rs):
    """A(m+1), B(m) match counts (Chebyshev distance <= r) at every tolerance in sorted rs.

    KD-tree pair counting (scipy cKDTree.count_neighbors with an array of radii);
    identical counts to the former O(N^2) loop, which made the full Nagoya window
    intractable and forced a 2000-beat cap.
    """
    from scipy.spatial import cKDTree
    N = len(x); nr = len(rs); A = np.zeros(nr); B = np.zeros(nr)
    for mm, acc in ((m, B), (m + 1, A)):
        M = N - mm + 1
        if M < 2: continue
        emb = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(x, mm)[:M])
        t = cKDTree(emb)
        c = t.count_neighbors(t, np.asarray(rs, float), p=np.inf)
        acc += (np.asarray(c, float) - M) / 2.0
    return A, B

def rcmse_multi(x, scales, rs, m=2):
    out = np.full((len(rs), len(scales)), np.nan)
    for si, s in enumerate(scales):
        if s == 1:
            A, B = counts_multi(x, m, rs)
        else:
            A = np.zeros(len(rs)); B = np.zeros(len(rs))
            for k in range(s):
                y = x[k:]; L = (len(y)//s)*s
                if L < s*(m+2): continue
                cg = y[:L].reshape(-1, s).mean(axis=1)
                a, b = counts_multi(cg, m, rs); A += a; B += b
        ok = (A > 0) & (B > 0)
        out[ok, si] = -np.log(A[ok]/B[ok])
    return out

# HRV-Complexity root, derived from this file (scripts/ -> public_release/ -> Multicenter/ -> root)
MC=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE=os.path.dirname(os.path.dirname(MC))
CAP=None   # no cap: full 18:00-22:00 window (KD-tree counts)
def load(name):
    recs=[]
    if name=='CETRAM':
        for g in ['Control','PD']:
            for p in sorted(glob.glob(f'{BASE}/CETRAM/public_release/results/cleaned_detections/{g}/*_cleaned.csv')):
                s=os.path.basename(p).replace('_cleaned.csv','')
                x=np.diff(pd.read_csv(p)['sample'].values.astype(float)); x=x[np.isfinite(x)]
                # Same 300-2000 ms guard as compute_length_ladder.py and the Methods.
                recs.append((s,g,x[(x>300.0)&(x<2000.0)]))
    elif name=='Cruces':
        grp=dict(pd.read_csv(f'{MC}/data/spain_mse.csv')[['Subject','Group']].drop_duplicates().values)
        for p in sorted(glob.glob(f'{BASE}/Cruces/public_release/data/processed/RRi/*.csv')):
            s=os.path.basename(p).replace('.csv','')
            if grp.get(s) not in ('Control','PD'): continue
            x=np.loadtxt(p).astype(float); x=x[np.isfinite(x)]; recs.append((s,grp[s],x[(x>300.0)&(x<2000.0)]))
    else:
        meta=pd.read_csv(f'{BASE}/Nagoya/public_release/data/metadata/metadata.csv')
        grp=dict(pd.read_csv(f'{MC}/data/japan_window_mse.csv')[['Subject','Group']].drop_duplicates().values)
        for _,inf in meta.iterrows():
            s=str(inf['Subject_ID']).strip()
            if grp.get(s) not in ('Control','PD'): continue
            f=f'{BASE}/Nagoya/public_release/data/processed_rri/{s}_RRi.txt'
            if not os.path.exists(f): continue
            a=np.loadtxt(f); t,r=a[:,0],a[:,1]*1000.0
            try:
                st=datetime.strptime(str(inf['Start_Time']).strip(),'%H:%M:%S'); off=st.hour*3600+st.minute*60+st.second
            except Exception: off=9*3600
            ab=(t+off)%86400; msk=(ab>=64800)&(ab<79200)   # 18:00-22:00, the pre-specified window
            x=r[msk]; x=x[np.isfinite(x)]
            # Physiological filter, matching compute_japan_window_features.py
            # (RRI_MIN_S=0.3, RRI_MAX_S=2.0). The processed_rri files encode data
            # GAPS as single huge "intervals" -- up to 211 000 ms -- which would
            # otherwise inflate SD by >50x and corrupt r = k x SD.
            x=x[(x>300.0)&(x<2000.0)]
            if len(x)<200: continue
            recs.append((s,grp[s],x if CAP is None else x[:CAP]))
    return recs

KS=np.array([0.05,0.10,0.125,0.15,0.175,0.20,0.25,0.30,0.35,0.40,0.50])
RABS=np.array([4.,6.,8.,10.,12.5,15.,20.,25.,30.,40.,50.,60.])
SC=list(range(1,6)); MS=[1,2,3]; rows=[]
import sys
COHORTS=[a for a in sys.argv[1:] if a in ('CETRAM','Cruces','Nagoya')] or ['CETRAM','Cruces','Nagoya']
OUTNAME='r_sweep.csv' if len(COHORTS)==3 else 'r_sweep_'+'_'.join(COHORTS)+'.csv'
# Resumable: one part file per (cohort, subject) under data/_rsweep_parts/; time-limited
# runs (--max-seconds N) stop cleanly and continue on the next call. The final table is
# written once every subject of the requested cohorts is done; with a single cohort the
# rows of that cohort replace its rows in data/r_sweep.csv.
import time
DATA=os.path.join(MC,'data'); PARTS=os.path.join(DATA,'_rsweep_parts'); os.makedirs(PARTS,exist_ok=True)
SH=tuple(int(v) for v in sys.argv[sys.argv.index('--shard')+1].split('/')) if '--shard' in sys.argv else (0,1)
TMAX=float(sys.argv[sys.argv.index('--max-seconds')+1]) if '--max-seconds' in sys.argv else 1e12
t0=time.time(); complete=True
for coh in COHORTS:
    recs=load(coh)
    print(f"{coh}: n={len(recs)} medianN={int(np.median([len(x) for _,_,x in recs]))}",flush=True)
    for j,(s,g,x) in enumerate(recs):
        part=os.path.join(PARTS,f'{coh}_{s}.csv')
        if os.path.exists(part) or j % SH[1] != SH[0]: continue
        if time.time()-t0>TMAX: complete=False; break
        rows=[]
        sd=np.std(x,ddof=1); h=len(x)//2
        rs_all=np.concatenate([KS*sd,RABS]); order=np.argsort(rs_all); rs=rs_all[order]
        inv=np.empty_like(order); inv[order]=np.arange(len(order))
        for mm in MS:
            cur=rcmse_multi(x,SC,rs,m=mm)
            c1=rcmse_multi(x[:h],SC,rs,m=mm); c2=rcmse_multi(x[h:],SC,rs,m=mm)
            for i,lab in enumerate(list(KS)+list(RABS)):
                p=inv[i]
                f=lambda a: np.trapezoid(a[p],SC)/5 if np.all(np.isfinite(a[p])) else np.nan
                rows.append(dict(Cohort=coh,Subject=s,Group=g,m=mm,
                    mode='k' if i<len(KS) else 'abs',
                    level=lab,r_ms=rs[p],SD=sd,N=len(x),cx=f(cur),h1=f(c1),h2=f(c2)))
        pd.DataFrame(rows).to_csv(part,index=False)
        print(f"   {coh} {s} {time.time()-t0:.0f}s",flush=True)
    if not complete: break
if SH[1] > 1 and complete:
    complete = all(os.path.exists(os.path.join(PARTS,f'{c}_{s}.csv')) for c in COHORTS for s,_,_ in load(c))
    if not complete: print('shard done; others pending')
if not complete:
    print("PAUSE: re-run to continue")
else:
    new=pd.concat([pd.read_csv(os.path.join(PARTS,f'{c}_{s}.csv')) for c in COHORTS for s,_,_ in load(c)],ignore_index=True)
    out=os.path.join(DATA,'r_sweep.csv')
    if len(COHORTS)<3 and os.path.exists(out):
        old=pd.read_csv(out); new=pd.concat([old[~old.Cohort.isin(COHORTS)],new],ignore_index=True)
    new.to_csv(out,index=False)
    print("DONE ->",out)
