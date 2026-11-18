import os, sys, glob, numpy as np, pandas as pd, neurokit2 as nk, warnings; warnings.filterwarnings('ignore')
from scipy.signal import welch
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ppg')); from ppg_lib import curve
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# --- (A2) where are the long holds? slope at hold positions vs typical upstroke slope
hs, us = [], []
for f in sorted(glob.glob(f'{S}/fid/ppg_traces/*.csv'))[:20]:
    x = np.loadtxt(f); ch = np.flatnonzero(np.diff(x) != 0) + 1; runs = np.diff(ch)
    ph = np.bincount(ch % 5, minlength=5).argmax(); idx = np.arange(ph, len(x), 5)
    r = CubicSpline(idx, x[idx])(np.arange(len(x))); d = np.gradient(nk.ppg_clean(r, sampling_rate=500))
    thr = np.percentile(np.abs(d), 90)
    pos = ch[:-1][runs >= 9] + 4
    hs.append(np.mean(np.abs(d[pos]) > thr)); us.append(0.10)
print("(A2) share of long holds located on steep segments (top-10%% |slope|): %.3f (chance 0.10) -> %s" %
      (np.mean(hs), "holds sit on flat parts: natural integer flatness, not dropped samples" if np.mean(hs) < 0.05 else "holds hit the upstroke: likely dropped samples"))
# --- (B) curves to tau=20, pooled, no labels: Cruces (new pipeline series) vs ECG at the same length
ser = {}
for f in glob.glob(f'{S}/verify/out_rri/*.csv'): ser[('Cruces', os.path.basename(f)[:-4])] = np.loadtxt(f)
for f in glob.glob(f'{S}/ppg/cetram/*/*_cleaned.csv'):
    x = np.diff(pd.read_csv(f)['sample'].values.astype(float)); x = x[(x >= 300) & (x <= 2000)]
    ser[('CETRAM full', os.path.basename(f))] = x; ser[('CETRAM 500 beats', os.path.basename(f))] = x[:500]
for f in sorted(glob.glob(f'{S}/dur/data/processed_rri/*_RRi.txt')):
    d = np.loadtxt(f); x = d[:, 1]*1000; x = x[(x >= 300) & (x <= 2000)]; ser[('Nagoya 500 beats', os.path.basename(f))] = x[20000:20500] if len(x) > 20500 else x[:500]
rows = []
for (c, s), x in ser.items():
    cu = curve(x, T=20); rows.append(dict(cohort=c, s=s, N=len(x), **{f't{k+1}': v for k, v in enumerate(cu)}))
C = pd.DataFrame(rows); C.to_csv(f'{S}/verify/curves20.csv', index=False)
print("\n(B) median rcMSE by scale (r = 0.2 SD fixed); [n undefined at that scale]")
print(f"{'cohort':18s} N   " + " ".join(f"t{k:<5d}" for k in (1, 2, 3, 4, 5, 8, 12, 16, 20)) + "  falling 1->5   between-subject SD t1 / t5 / t20")
for c, g in C.groupby('cohort', sort=False):
    meds = " ".join(f"{g[f't{k}'].median():.2f} " for k in (1, 2, 3, 4, 5, 8, 12, 16, 20))
    und = g[[f't{k}' for k in range(1, 21)]].isna().sum().sum()
    print(f"{c:18s} {int(g.N.median()):4d} {meds} {100*(g.t5 < g.t1).mean():5.0f}%   {g.t1.std():.2f} / {g.t5.std():.2f} / {g.t20.std():.2f}  undefined values: {und}")
# --- (C) where is the variance? beat-domain spectrum, pooled median share per band (cycles/beat)
print("\n(C) median share of detrended interval variance by beat-frequency band (cycles/beat)")
bands = [(0.02, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
for c in ('Cruces', 'CETRAM 500 beats', 'Nagoya 500 beats'):
    sh = []
    for (cc, s), x in ser.items():
        if cc != c: continue
        y = x - pd.Series(x).rolling(31, center=True, min_periods=1).mean().values
        f, p = welch(y, fs=1.0, nperseg=min(128, len(y)//2)); tot = p[f > 0.02].sum()
        sh.append([p[(f >= a) & (f < b)].sum()/tot for a, b in bands])
    sh = np.median(np.array(sh), axis=0); print(f"  {c:18s} " + "  ".join(f"{a:.2f}-{b:.2f}: {v:.2f}" for (a, b), v in zip(bands, sh)))
