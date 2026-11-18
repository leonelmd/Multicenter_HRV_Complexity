"""Label-blind: is the excess short-term variability white (timing noise) or band-limited (respiration)?
hf_frac = share of IBI variance at 0.35-0.5 cycles/beat (alternating beat-to-beat changes). Breathing at
<=20/min and HR >= 57 bpm sits below 0.35 cycles/beat, so that band is mostly noise; white noise would put
~30% of the variance there. resp_peak = spectral peak frequency in 0.1-0.35 cycles/beat converted to breaths/min.
Also NeuroKit2 template-matching PPG quality (mean over beats)."""
import os, glob, numpy as np, pandas as pd, neurokit2 as nk
from scipy.signal import welch
W = os.path.dirname(os.path.abspath(__file__)); S = os.path.dirname(W)
beats = np.load(f'{W}/beats.npy', allow_pickle=True).item(); D = pd.read_csv(f'{W}/diagnostics_per_subject.csv')
def clean(x):
    x = x[(x > 300) & (x < 2000)]; med = pd.Series(x).rolling(11, center=True, min_periods=1).median().values
    return x[np.abs(x - med) <= 0.2*med]
def spec(x):
    x = clean(x); y = x - pd.Series(x).rolling(31, center=True, min_periods=1).mean().values  # remove slow drift
    f, p = welch(y, fs=1.0, nperseg=min(128, len(y)//2)); tot = p[f > 0.02].sum()
    band = (f >= 0.1) & (f < 0.35); fpk = f[band][np.argmax(p[band])]
    return p[f >= 0.35].sum()/tot, fpk*60000/np.mean(x)
rows = []
for s, b in beats.items():
    x = np.diff(b['maxd1']); x = x[np.isfinite(x)]; h, br = spec(x)
    sig = np.loadtxt(f'{W}/ppg_traces/{s}.csv'); c = nk.ppg_clean(sig, sampling_rate=500)
    qy = np.nanmean(nk.ppg_quality(c, peaks=(b['apex_raw']/2).astype(int), sampling_rate=500, method='templatematch'))
    rows.append(dict(src='Cruces PPG (maxd1)', s=s, hf_frac=h, resp_bpm=br, quality=qy))
for f in glob.glob(f'{S}/ppg/cetram/*/*_cleaned.csv'):
    h, br = spec(np.diff(pd.read_csv(f)['sample'].dropna().values.astype(float))); rows.append(dict(src='CETRAM ECG', s=os.path.basename(f)[:-12], hf_frac=h, resp_bpm=br))
R = pd.DataFrame(rows)
C = D[(D.src == 'Cruces PPG') & (D.fid == 'maxd1')][['s', 'ratio', 'lag1', 'RMSSD', 'SDNN']].merge(R[R.src.str.startswith('Cruces')], on='s')
q = lambda v: f"{np.nanmedian(v):.3f} [{np.nanpercentile(v,25):.3f}-{np.nanpercentile(v,75):.3f}]"
for src, g in R.groupby('src'):
    print(f"{src:20s} n={len(g):2d}  HF fraction (0.35-0.5 cyc/beat) {q(g.hf_frac)}   resp peak {np.nanmedian(g.resp_bpm):.1f}/min")
print(f"\nCruces: Spearman with RMSSD/SDNN ->  HF fraction {C[['ratio','hf_frac']].corr('spearman').iloc[0,1]:+.2f}   PPG quality {C[['ratio','quality']].corr('spearman').iloc[0,1]:+.2f}")
thr = np.nanpercentile(R[R.src=='CETRAM ECG'].hf_frac, 95)
C['noisy'] = C.hf_frac > thr
print(f"Cruces recordings above the CETRAM 95th percentile of HF fraction ({thr:.3f}): {C.noisy.sum()}/{len(C)}")
print(C.sort_values('hf_frac', ascending=False)[['s','hf_frac','ratio','lag1','RMSSD','SDNN','quality','resp_bpm']].head(15).round(3).to_string(index=False))
print("\nremaining Cruces (not flagged): RMSSD/SDNN %s  lag-1 %s" % (q(C[~C.noisy].ratio), q(C[~C.noisy].lag1)))
C.to_csv(f'{W}/cruces_noise_screen.csv', index=False)
