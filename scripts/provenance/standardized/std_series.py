"""Standardized series, steps 1-4, label-blind. Writes per-recording series + screening table."""
import os, glob, numpy as np, pandas as pd, neurokit2 as nk, warnings; warnings.filterwarnings('ignore')
from scipy.signal import welch
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(S, 'std')
def lt(t_ms):
    """Lipponen-Tarvainen on beat times (ms), 0.1-ms resolution; returns corrected times, n corrected."""
    pk = np.round(np.asarray(t_ms)*10).astype(np.int64); pk = pk - pk[0]
    info, pk2 = nk.signal_fixpeaks(pk, sampling_rate=10000, method='Kubios', iterative=True)
    n = sum(len(info[k]) for k in ('ectopic', 'missed', 'extra', 'longshort'))
    return np.asarray(pk2)/10.0 + t_ms[0], n
def guard(x): return x[(x >= 300) & (x <= 2000)]
def hf_frac(x):
    y = x - pd.Series(x).rolling(31, center=True, min_periods=1).mean().values
    f, p = welch(y, fs=1.0, nperseg=min(128, len(y)//2)); return p[f >= 0.35].sum()/p[f > 0.02].sum()
series, rows = {}, []
# CETRAM: already LT-corrected in cleaned_detections
keep = set(pd.read_csv('/mnt/user-data/uploads/Multicenter/public_release/data/chile_mse.csv').Subject)
for f in glob.glob(f'{S}/ppg/cetram/*/*_cleaned.csv'):
    s = os.path.basename(f)[:-12]
    if s not in keep: continue
    x = guard(np.diff(pd.read_csv(f)['sample'].dropna().values.astype(float)))
    series[('CETRAM', s)] = x; rows.append(dict(cohort='CETRAM', s=s, n=len(x), corrected=np.nan, hf=hf_frac(x)))
# Cruces: max-slope fiducial + LT
beats = np.load(f'{S}/fid/beats.npy', allow_pickle=True).item()
for s, b in beats.items():
    t = b['maxd1']; t = t[np.isfinite(t)]; t2, n = lt(t); x = guard(np.diff(t2))
    series[('Cruces', s)] = x; rows.append(dict(cohort='Cruces', s=s, n=len(x), corrected=n, hf=hf_frac(x)))
# Nagoya 18:00-22:00: segments split at gaps (>2 s between consecutive beat times beyond the RR), LT per segment
meta = pd.read_csv(f'{S}/dur/data/metadata/metadata.csv')
st = {r.Subject_ID: sum(int(v)*m for v, m in zip(str(r.Start_Time).split(':'), (3600, 60, 1))) for r in meta.itertuples()}
win = set(pd.read_csv('/mnt/user-data/uploads/Multicenter/public_release/data/japan_15min_windows.csv').Subject)
for f in sorted(glob.glob(f'{S}/dur/data/processed_rri/*_RRi.txt')):
    s = os.path.basename(f)[:-8]; d = np.loadtxt(f); t, rr = d[:, 0]*1000, d[:, 1]*1000
    clk = ((t/1000 + st[s]) % 86400)/3600; sel = (clk >= 18) & (clk < 22)
    if sel.sum() < 1000: continue
    tt, r = t[sel], rr[sel]
    beat_t = tt + r            # time of the beat that closes each interval
    brk = np.where((r > 2000) | (r < 300))[0]
    xs, ncor = [], 0; edges = np.r_[-1, brk, len(r)]
    for a, b in zip(edges[:-1], edges[1:]):
        seg = beat_t[a+1:b]
        if len(seg) < 30: continue
        t2, n = lt(seg); ncor += n; xs.append(guard(np.diff(t2)))
    x = np.concatenate(xs); series[('Nagoya', s)] = x
    rows.append(dict(cohort='Nagoya', s=s, n=len(x), corrected=ncor, hf=hf_frac(x)))
R = pd.DataFrame(rows); R['exclude'] = R.hf >= 0.30
R.to_csv(f'{OUT}/screening.csv', index=False); np.save(f'{OUT}/series.npy', series, allow_pickle=True)
for c, g in R.groupby('cohort', sort=False):
    print(f"{c:7s} n={len(g):2d}  LT-corrected beats median {g.corrected.median():.0f} (max {g.corrected.max():.0f})   "
          f"alternating-variance share median {g.hf.median():.3f}   excluded (>=0.30): {g.exclude.sum()}  {sorted(g[g.exclude].s)}")
