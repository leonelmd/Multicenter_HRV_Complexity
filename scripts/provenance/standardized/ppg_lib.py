import os, glob, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
W = os.path.dirname(os.path.abspath(__file__)); U = '/mnt/user-data/uploads/Multicenter/public_release'
def AB(s, m, r):
    n = len(s) - m
    if n < 3: return 0.0, 0.0
    X = s[np.arange(n)[:, None] + np.arange(m)[None, :]].astype(np.float32)
    d = np.abs(X[:, None, :] - X[None, :, :]).max(2); xm = s[np.arange(n) + m].astype(np.float32)
    d1 = np.maximum(d, np.abs(xm[:, None] - xm[None, :])); iu = np.triu_indices(n, 1)
    return float((d1[iu] <= r).sum()), float((d[iu] <= r).sum())
def curve(x, m=2, T=5):
    r = 0.2*np.std(x, ddof=1); out = []
    for t in range(1, T+1):
        A = B = 0.0
        for k in range(t):
            L = (len(x)-k)//t
            if L <= m: continue
            a, b = AB(x[k:k+L*t].reshape(L, t).mean(1), m, r); A += a; B += b
        out.append(-np.log(A/B) if A > 0 and B > 0 else np.nan)
    return np.array(out)
nauc = lambda c: np.nan if not np.isfinite(c).all() else float(np.trapezoid(c, np.arange(1, 6))/5)
def med3(x, lo=4.0, hi=18.0):   # Chang & Yu adaptive median-of-3
    x = np.asarray(x, float); m = x.copy()
    m[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
    a = np.clip((np.abs(x-m)-lo)/(hi-lo), 0, 1); return (1-a)*x + a*m
def clean(x):  # same rule as CETRAM's pipeline: 300-2000 ms and within 20% of local median(11), delete
    x = x[(x > 300) & (x < 2000)]
    med = pd.Series(x).rolling(11, center=True, min_periods=1).median().values
    return x[np.abs(x-med) <= 0.2*med]
def auc(c, p):
    c = np.asarray(c, float); p = np.asarray(p, float); c = c[~np.isnan(c)]; p = p[~np.isnan(p)]
    u = mannwhitneyu(c, p); return u.statistic/(len(c)*len(p)), u.pvalue
def diag(x):
    d = np.diff(x); xc = x - x.mean()
    return dict(n=len(x), SDNN=np.std(x, ddof=1), RMSSD=np.sqrt(np.mean(d**2)),
                ratio=np.sqrt(np.mean(d**2))/np.std(x, ddof=1), lag1=np.corrcoef(x[:-1], x[1:])[0,1])
