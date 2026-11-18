import os, sys, glob, numpy as np, pandas as pd
from scipy.stats import kurtosis
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ppg')); from ppg_lib import curve
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
beats = np.load(f'{S}/fid/beats.npy', allow_pickle=True).item()
def g(x): return x[(x >= 300) & (x <= 2000)]
# 1. heavy tails of successive differences: Cruces vs ECG
k = {'Cruces maxd1': [kurtosis(np.diff(g(np.diff(b['maxd1'][np.isfinite(b['maxd1'])])))) for b in beats.values()]}
k['CETRAM'] = [kurtosis(np.diff(g(np.diff(pd.read_csv(f)['sample'].values.astype(float))[:500]))) for f in glob.glob(f'{S}/ppg/cetram/*/*_cleaned.csv')]
print("excess kurtosis of successive differences (median): " + "  ".join(f"{c} {np.median(v):.1f}" for c, v in k.items()))
# 2. do fiducials jump? beat-to-beat delay between fiducials on the same pulse: bimodality / big jumps
rows = []
for s, b in beats.items():
    for f2 in ('mid', 'tangent', 'apex'):
        d = b[f2] - b['maxd1']; d = d[np.isfinite(d)]; dd = d - np.median(d)
        rows.append(dict(s=s, pair=f'{f2}-maxd1', pct_jump_gt8ms=100*np.mean(np.abs(dd) > 8), mad=np.median(np.abs(dd))))
J = pd.DataFrame(rows)
print("\nshare of beats where two fiducials on the SAME pulse disagree by > 8 ms from their usual offset (median, IQR):")
for p, gg in J.groupby('pair'): print(f"   {p:14s} {gg.pct_jump_gt8ms.median():5.1f}% [{gg.pct_jump_gt8ms.quantile(.25):.1f}-{gg.pct_jump_gt8ms.quantile(.75):.1f}]")
# 3. curve shape under each fiducial (same beats, guard only)
print("\nshare of falling curves (t5 < t1) and median t1 / t5 by fiducial:")
for fid in ('apex', 'maxd1', 'mid', 'tangent'):
    sl = []
    for b in beats.values():
        t = b[fid]; x = g(np.diff(t[np.isfinite(t)])); c = curve(x); sl.append((c[0], c[4]))
    sl = np.array(sl); print(f"   {fid:8s} falling {100*np.mean(sl[:,1] < sl[:,0]):3.0f}%   t1 {np.median(sl[:,0]):.2f}  t5 {np.median(sl[:,1]):.2f}")
