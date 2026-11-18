"""Label-blind diagnostics of each fiducial series (no group labels read)."""
import os, glob, numpy as np, pandas as pd
W = os.path.dirname(os.path.abspath(__file__)); S = os.path.dirname(W)
beats = np.load(f'{W}/beats.npy', allow_pickle=True).item()
def clean(x):
    x = x[(x > 300) & (x < 2000)]
    med = pd.Series(x).rolling(11, center=True, min_periods=1).median().values
    return x[np.abs(x - med) <= 0.2*med]
def sampen(x, m=2):
    r = 0.2*np.std(x, ddof=1); n = len(x) - m
    X = np.lib.stride_tricks.sliding_window_view(x, m)[:n].astype(np.float32)
    d = np.abs(X[:, None, :] - X[None, :, :]).max(2); xm = x[m:m+n].astype(np.float32)
    d1 = np.maximum(d, np.abs(xm[:, None] - xm[None, :])); iu = np.triu_indices(n, 1)
    A, B = (d1[iu] <= r).sum(), (d[iu] <= r).sum(); return -np.log(A/B) if A and B else np.nan
def stats(x):
    xc = clean(x); d = np.diff(xc)
    return dict(n=len(x), pct_flag=100*(1 - len(xc)/len(x)), SDNN=np.std(xc, ddof=1), RMSSD=np.sqrt(np.mean(d**2)),
                ratio=np.sqrt(np.mean(d**2))/np.std(xc, ddof=1), lag1=np.corrcoef(xc[:-1], xc[1:])[0, 1], SampEn1=sampen(xc))
def ibi(t):
    i = np.diff(t); ok = np.isfinite(i); return i[ok]
# --- sanity: apex as detected reproduces the stored RRi exactly
mism = 0
for s, b in beats.items():
    stored = np.loadtxt(f'{S}/ppg/cruces/RRi/{s}.csv', delimiter=',', ndmin=1)
    mism += not (len(stored) == len(b['apex_raw'])-1 and np.allclose(stored, np.diff(b['apex_raw'])))
print(f"sanity: re-detected apex reproduces stored Cruces RRi in {len(beats)-mism}/{len(beats)} recordings")
rows = []
for s, b in beats.items():
    for k in ('apex_raw', 'apex', 'maxd1', 'mid', 'tangent', 'foot'):
        rows.append(dict(src='Cruces PPG', fid=k, s=s, **stats(ibi(b[k]))))
    for k in ('apex', 'mid', 'tangent', 'foot'):   # beat-to-beat spread of the delay between two points on the same pulse
        dl = b[k] - b['maxd1']; dl = dl[np.isfinite(dl)]
        rows.append(dict(src='delay', fid=f'{k}-maxd1', s=s, n=len(dl), SDNN=np.median(dl), RMSSD=np.std(dl, ddof=1)))
# --- ECG references
for f in glob.glob(f'{S}/ppg/cetram/*/*_cleaned.csv'):
    pk = pd.read_csv(f)['sample'].dropna().values.astype(float); rows.append(dict(src='CETRAM ECG', fid='R', s=os.path.basename(f), **stats(np.diff(pk))))
meta = pd.read_csv(f'{S}/dur/data/metadata/metadata.csv'); st = {r.Subject_ID: sum(int(v)*m for v, m in zip(str(r.Start_Time).split(':'), (3600, 60, 1))) for r in meta.itertuples()}
for f in glob.glob(f'{S}/dur/data/processed_rri/*_RRi.txt'):
    s = os.path.basename(f)[:-8]; d = np.loadtxt(f); t, x = d[:, 0], d[:, 1]*1000; clk = ((t + st[s]) % 86400)/3600
    # one 7.5-min segment (Cruces length), first one starting 10:00-12:00, no gaps > 3 s
    for a in np.where((clk >= 10) & (clk < 12))[0][::200]:
        sel = (t >= t[a]) & (t < t[a] + 450)
        if sel.sum() > 300 and np.max(np.diff(t[sel])) < 3: rows.append(dict(src='Nagoya ECG strap', fid='R', s=s, **stats(x[sel]))); break
D = pd.DataFrame(rows); D.to_csv(f'{W}/diagnostics_per_subject.csv', index=False)
q = lambda v: f"{np.nanmedian(v):.2f} [{np.nanpercentile(v,25):.2f}-{np.nanpercentile(v,75):.2f}]"
print("\nmedian [IQR] across recordings; series cleaned with the CETRAM rule")
print(f"{'source':17s} {'fiducial':9s} {'n':>3s} {'%flagged':>9s} {'RMSSD/SDNN':>19s} {'lag-1 autocorr':>19s} {'SampEn tau=1':>19s} {'RMSSD ms':>7s} {'SDNN ms':>7s}")
for (src, fid), g in D[D.src != 'delay'].groupby(['src', 'fid'], sort=False):
    print(f"{src:17s} {fid:9s} {len(g):3d} {g.pct_flag.median():8.1f}% {q(g.ratio):>19s} {q(g.lag1):>19s} {q(g.SampEn1):>19s} {g.RMSSD.median():7.1f} {g.SDNN.median():7.1f}")
print("\ndelay between fiducials on the same pulse: median delay (ms) and its beat-to-beat SD (ms)")
for fid, g in D[D.src == 'delay'].groupby('fid'):
    print(f"   {fid:14s} delay {g.SDNN.median():6.1f} ms   beat-to-beat SD {q(g.RMSSD)} ms")
