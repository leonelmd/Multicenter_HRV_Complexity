import os, glob, numpy as np, pandas as pd, neurokit2 as nk, warnings; warnings.filterwarnings('ignore')
from scipy.signal import welch
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
beats = np.load(f'{S}/fid/beats.npy', allow_pickle=True).item()
# 1. timing of spike beats: autocorrelation of the spike train (1-s bins) pooled across recordings
ac_sum = np.zeros(61); n_ev = 0; ev_all = []
for s, b in beats.items():
    t = b['mid']; t = t[np.isfinite(t)]; x = np.diff(t); d = np.diff(x); tm = t[2:]/1000
    mad = np.median(np.abs(d - np.median(d))); ev = tm[np.abs(d - np.median(d)) > 5*1.4826*mad]
    n_ev += len(ev); ev_all.append(ev)
    tr = np.zeros(int(tm.max()*10)+1); tr[(ev*10).astype(int)] = 1        # 100-ms bins
    tr = tr - tr.mean(); ac = np.correlate(tr, tr, 'full')[len(tr)-1:len(tr)-1+601]
    ac_sum += ac[::10][:61]/max(ac[0], 1e-9)
print(f"spike beats (>5 robust SD in successive difference): {n_ev} over {len(beats)} recordings ({n_ev/len(beats):.1f} per recording)")
ac = ac_sum/len(beats); top = np.argsort(ac[1:])[::-1][:5] + 1
print("pooled spike-train autocorrelation, top lags (s):", [(int(l), round(float(ac[l]), 3)) for l in top], "  median |ac| lags 1-60:", round(float(np.median(np.abs(ac[1:]))), 3))
# 2. narrow spectral lines in the raw PPG (not at heart-rate harmonics): pooled normalized spectrum
P = None
for f in sorted(glob.glob(f'{S}/fid/ppg_traces/*.csv'))[:52]:
    x = np.loadtxt(f); fr, p = welch(x - x.mean(), fs=500, nperseg=2**15); p = p/np.median(p[(fr > 0.5) & (fr < 20)])
    P = p if P is None else P + p
P = P/52; m = (fr > 0.05) & (fr < 25)
# line-ness: power relative to local median over +-0.3 Hz
loc = pd.Series(P).rolling(int(0.6/(fr[1]-fr[0])), center=True, min_periods=1).median().values
ratio = P/loc; cand = np.where(m & (ratio > 8))[0]
print("narrow lines in pooled raw-PPG spectrum (freq Hz, peak/local-median):", [(round(float(fr[i]), 3), round(float(ratio[i]), 1)) for i in cand[np.argsort(ratio[cand])[::-1][:10]]])
