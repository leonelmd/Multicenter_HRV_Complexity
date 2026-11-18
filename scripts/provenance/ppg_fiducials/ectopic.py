"""Label-blind. Apply the Lipponen & Tarvainen (2019) correction, NeuroKit2 signal_fixpeaks(method='Kubios'),
to the Cruces beat times (maxd1 fiducial, 0.1 ms resolution), then recompute the diagnostics."""
import os, numpy as np, pandas as pd, neurokit2 as nk, warnings; warnings.filterwarnings('ignore')
from scipy.signal import welch
W = os.path.dirname(os.path.abspath(__file__)); beats = np.load(f'{W}/beats.npy', allow_pickle=True).item()
def clean(x):
    x = x[(x > 300) & (x < 2000)]; med = pd.Series(x).rolling(11, center=True, min_periods=1).median().values
    return x[np.abs(x - med) <= 0.2*med]
def diag(x):
    x = clean(x); d = np.diff(x); y = x - pd.Series(x).rolling(31, center=True, min_periods=1).mean().values
    f, p = welch(y, fs=1.0, nperseg=min(128, len(y)//2))
    return dict(ratio=np.sqrt(np.mean(d**2))/np.std(x, ddof=1), lag1=np.corrcoef(x[:-1], x[1:])[0, 1], hf=p[f >= 0.35].sum()/p[f > 0.02].sum())
rows, fixed = [], {}
for s, b in beats.items():
    t = b['maxd1']; t = t[np.isfinite(t)]; pk = np.round(t*10).astype(int)          # 0.1-ms units
    info, pk2 = nk.signal_fixpeaks(pk, sampling_rate=10000, method='Kubios', iterative=True)
    n_art = sum(len(info[k]) for k in ('ectopic', 'missed', 'extra', 'longshort'))
    t2 = np.asarray(pk2)/10.0; fixed[s] = t2
    rows.append(dict(s=s, beats=len(t), corrected=n_art, pct=100*n_art/len(t),
                     **{k+'_before': v for k, v in diag(np.diff(t)).items()}, **{k+'_after': v for k, v in diag(np.diff(t2)).items()}))
R = pd.DataFrame(rows); R.to_csv(f'{W}/ectopic_correction.csv', index=False); np.save(f'{W}/beats_fixed.npy', fixed, allow_pickle=True)
q = lambda v: f"{np.nanmedian(v):.3f} [{np.nanpercentile(v,25):.3f}-{np.nanpercentile(v,75):.3f}]"
print(f"beats corrected per recording: median {R.corrected.median():.0f} ({R.pct.median():.1f}%), max {R.corrected.max()} ({R.pct.max():.1f}%); recordings with any correction {(R.corrected>0).sum()}/{len(R)}")
for k in ('ratio', 'lag1', 'hf'):
    print(f"  {k:6s} before {q(R[k+'_before'])}   after {q(R[k+'_after'])}")
print("  references: CETRAM ECG ratio 0.57 [0.45-0.73], lag-1 0.84 [0.73-0.90], hf 0.046 [0.029-0.105]; Nagoya ratio 0.47, lag-1 0.89")
print("\nfour flagged recordings:"); print(R[R.s.isin(['E14','C07','C19','E35'])][['s','corrected','pct','ratio_before','ratio_after','lag1_before','lag1_after','hf_before','hf_after']].round(3).to_string(index=False))
print(f"\nrecordings still above the CETRAM 95th pct of hf (0.287) after correction: {(R.hf_after > 0.287).sum()}")
