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
# NOTE: Nagoya is capped at CAP beats to keep the O(N^2) sweep tractable. The SHAPE of
# the r-response survives the cap, but absolute Nagoya AUCs are NOT comparable to the
# published full-length values -- the effect grows with N (1000: 0.599, 2000: 0.655,
# 4000: 0.681, full ~15900: 0.867).
#
# Output: data/r_sweep.csv

import numpy as np, pandas as pd, glob, os
from datetime import datetime

def counts_multi(x, m, rs):
    """A(m+1), B(m) match counts at every tolerance in sorted rs. One pass, no sorting of distances."""
    N = len(x); nr = len(rs); A = np.zeros(nr); B = np.zeros(nr)
    for mm, acc in ((m, B), (m + 1, A)):
        M = N - mm + 1
        if M < 2: continue
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[:M]
        tot = np.zeros(nr + 1)
        for i in range(M - 1):
            d = np.max(np.abs(emb[i+1:] - emb[i]), axis=1)
            tot += np.bincount(np.searchsorted(rs, d, side='left'), minlength=nr + 1)
        acc += np.cumsum(tot)[:nr]
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

BASE='/sessions/keen-eloquent-noether/mnt/HRV-Complexity'; MC=f'{BASE}/Multicenter/public_release'
CAP=2000
def load(name):
    recs=[]
    if name=='CETRAM':
        for g in ['Control','PD']:
            for p in sorted(glob.glob(f'{BASE}/CETRAM/public_release/results/cleaned_detections/{g}/*_cleaned.csv')):
                s=os.path.basename(p).replace('_cleaned.csv','')
                x=np.diff(pd.read_csv(p)['sample'].values.astype(float)); recs.append((s,g,x[np.isfinite(x)]))
    elif name=='Cruces':
        grp=dict(pd.read_csv(f'{MC}/data/spain_mse.csv')[['Subject','Group']].drop_duplicates().values)
        for p in sorted(glob.glob(f'{BASE}/Cruces/data/processed/RRi/*.csv')):
            s=os.path.basename(p).replace('.csv','')
            if grp.get(s) not in ('Control','PD'): continue
            x=np.loadtxt(p).astype(float); recs.append((s,grp[s],x[np.isfinite(x)]))
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
            ab=(t+off)%86400; msk=(ab>=57600)&(ab<72000)
            x=r[msk]; x=x[np.isfinite(x)]
            # Physiological filter, matching compute_japan_window_features.py
            # (RRI_MIN_S=0.3, RRI_MAX_S=2.0). The processed_rri files encode data
            # GAPS as single huge "intervals" -- up to 211 000 ms -- which would
            # otherwise inflate SD by >50x and corrupt r = k x SD.
            x=x[(x>300.0)&(x<2000.0)]
            if len(x)<200: continue
            recs.append((s,grp[s],x[:CAP]))
    return recs

KS=np.array([0.05,0.10,0.125,0.15,0.175,0.20,0.25,0.30,0.35,0.40,0.50])
RABS=np.array([4.,6.,8.,10.,12.5,15.,20.,25.,30.,40.,50.,60.])
SC=list(range(1,6)); MS=[1,2,3]; rows=[]
for coh in ['CETRAM','Cruces','Nagoya']:
    recs=load(coh)
    print(f"{coh}: n={len(recs)} medianN={int(np.median([len(x) for _,_,x in recs]))}",flush=True)
    for j,(s,g,x) in enumerate(recs):
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
        if (j+1)%20==0: print(f"   {j+1}/{len(recs)}",flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data','r_sweep.csv'),index=False)
print("DONE")
