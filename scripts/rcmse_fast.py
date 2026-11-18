"""Fast rcMSE (Wu et al. 2014 refined composite MSE) using KD-tree pair counts.

Identical output to the O(N^2) implementation in compute_length_ladder.curve()
(r = k * SD of the original series, fixed across scales; Chebyshev distance <= r;
self-matches excluded). Used where series are too long for the pairwise matrix.
"""
import numpy as np
from scipy.spatial import cKDTree
def _AB(c,m,r):
    n=len(c)-m
    if n<3: return 0.0,0.0
    Xm=np.stack([c[i:i+n] for i in range(m)],1)
    Xm1=np.stack([c[i:i+n] for i in range(m+1)],1)
    tB=cKDTree(Xm); tA=cKDTree(Xm1)
    B=(tB.count_neighbors(tB,r,p=np.inf)-n)/2
    A=(tA.count_neighbors(tA,r,p=np.inf)-n)/2
    return float(A),float(B)
def rcmse(sig,nscales,m=2,k=0.2):
    sig=np.asarray(sig,float); N=len(sig); r=k*sig.std(ddof=1); out=[]
    for t in range(1,nscales+1):
        A=B=0.0
        for o in range(t):
            Lh=(N-o)//t
            if Lh<=m: continue
            c=sig[o:o+Lh*t].reshape(Lh,t).mean(1); a,b=_AB(c,m,r); A+=a;B+=b
        out.append(-np.log(A/B) if A>0 and B>0 else np.nan)
    return np.array(out)
