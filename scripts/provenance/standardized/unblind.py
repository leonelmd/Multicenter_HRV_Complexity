"""Group results under the pre-registered pipeline: CETRAM and Cruces."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, f'{S}/ppg'); 
from ppg_lib import curve, med3   # same rcMSE (reproduces manuscript values) and Chang & Yu smoother
ser = np.load(f'{S}/std/series.npy', allow_pickle=True).item(); R = pd.read_csv(f'{S}/std/screening.csv')
grp = pd.read_csv('/mnt/user-data/uploads/Multicenter/public_release/scratch/bundle2/spain_metrics.csv').set_index('Subject').Group.to_dict()
G = lambda c, s: ('Control' if s.startswith('C') else 'PD') if c == 'CETRAM' else grp[s]
def auc(c, p):
    u = mannwhitneyu(c, p); return u.statistic/(len(c)*len(p)), u.pvalue
def hr_adj(df):
    X = np.column_stack([np.ones(len(df)), (df.g == 'PD').astype(float), df.HR]); b = np.linalg.lstsq(X, df.cx.values, rcond=None)[0]
    res = df.cx.values - b[2]*df.HR.values; e = df.cx.values - X @ b
    XtXi = np.linalg.inv(X.T @ X); h = np.einsum('ij,jk,ik->i', X, XtXi, X)
    V = XtXi @ (X.T @ (X*(e**2/(1-h)**2)[:, None])) @ XtXi; from math import erfc, sqrt
    a, _ = auc(res[df.g == 'Control'], res[df.g == 'PD']); return a, erfc(abs(b[1]/np.sqrt(V[1, 1]))/sqrt(2))
rows = []
for (c, s), x in ser.items():
    if c == 'Nagoya': continue
    cur = curve(x); y = med3(x) if c == 'Cruces' else None
    rows.append(dict(cohort=c, s=s, g=G(c, s), cx=np.trapezoid(cur, np.arange(1, 6))/5, **{f'e{k+1}': v for k, v in enumerate(cur)},
                     cx_med3=(np.trapezoid(curve(y), np.arange(1, 6))/5 if y is not None else np.nan),
                     SDNN=np.std(x, ddof=1), RMSSD=np.sqrt(np.mean(np.diff(x)**2)), HR=60000/np.mean(x)))
D = pd.DataFrame(rows).merge(R[['cohort', 's', 'exclude']], on=['cohort', 's']); D.to_csv(f'{S}/std/unblinded_per_subject.csv', index=False)
for c in ('CETRAM', 'Cruces'):
    for lab, d in (('all recordings', D[D.cohort == c]), ('after rhythm screen', D[(D.cohort == c) & ~D.exclude])):
        C, P = d[d.g == 'Control'], d[d.g == 'PD']; a, p = auc(C.cx, P.cx); ah, ph = hr_adj(d)
        sd, rm = auc(C.SDNN, P.SDNN), auc(C.RMSSD, P.RMSSD)
        print(f"{c:6s} {lab:20s} C={len(C)} PD={len(P)} | complexity AUC {a:.3f} p={p:.3f} | HR-adj AUC {ah:.3f} p={ph:.3f} | SDNN {sd[0]:.3f} p={sd[1]:.3f} | RMSSD {rm[0]:.3f} p={rm[1]:.3f}")
        print("        per-scale AUC: " + "  ".join(f"tau{k} {auc(C[f'e{k}'], P[f'e{k}'])[0]:.3f}" for k in range(1, 6)))
        if c == 'Cruces':
            a3, p3 = auc(C.cx_med3, P.cx_med3); print(f"        sensitivity, + median-of-3: complexity AUC {a3:.3f} p={p3:.3f}")
print("\nexcluded by the rhythm screen:", D[D.exclude].groupby(['cohort', 'g']).s.apply(list).to_dict())
