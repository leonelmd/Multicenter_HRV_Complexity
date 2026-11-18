"""(A) Does the 100-Hz staircase degrade beat timing? Reconstruct the true 100-Hz samples (one value per
5-sample block, phase chosen where values change), cubic-spline them back to 500 Hz, rerun the identical
detector, and compare beat times with the staircase-based ones. Also count long holds (dropped samples)."""
import os, sys, glob, numpy as np, pandas as pd, neurokit2 as nk, warnings; warnings.filterwarnings('ignore')
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import prep
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
keep = set(open('/mnt/user-data/uploads/Multicenter/public_release/scratch/bundle3/analysed_subjects.txt').read().split())
rows = []
for f in sorted(glob.glob(f'{S}/fid/ppg_traces/*.csv')):
    s = os.path.basename(f)[:-4]
    if s not in keep: continue
    x = np.loadtxt(f)
    ch = np.flatnonzero(np.diff(x) != 0) + 1; runs = np.diff(ch)
    # phase of the 100-Hz grid = most common (change index mod 5)
    ph = np.bincount(ch % 5, minlength=5).argmax()
    idx = np.arange(ph, len(x), 5); y = x[idx]                     # true 100-Hz samples
    xr = CubicSpline(idx, y)(np.arange(len(x)))                    # band-limited-ish reconstruction at 500 Hz
    out = {}
    for lab, sig in (('stair', x), ('recon', xr)):
        c = nk.ppg_clean(sig, sampling_rate=500); pk = nk.ppg_findpeaks(c, sampling_rate=500, method='elgendi')['PPG_Peaks']
        out[lab] = prep.beat_times(c, pk, 500)
    a, b = out['stair'], out['recon']
    j = np.searchsorted(b, a); j = np.clip(j, 1, len(b)-1); j = np.where(np.abs(b[j-1]-a) < np.abs(b[j]-a), j-1, j)
    d = a - b[j]; d = d[np.abs(d) < 50]
    ia, ib = np.diff(a), np.diff(b)
    rows.append(dict(s=s, pattern='1-4' if np.median(runs) < 5 else '5', long_holds=int((runs >= 9).sum()), pct_time_in_long_holds=100*runs[runs >= 9].sum()/len(x),
                     beat_shift_sd=np.std(d), ibi_diff_sd=np.std(np.diff(a) [:min(len(ia),len(ib))] - np.diff(b)[:min(len(ia),len(ib))]) if len(ia)==len(ib) else np.nan,
                     rmssd_stair=np.sqrt(np.mean(np.diff(ia)**2)), rmssd_recon=np.sqrt(np.mean(np.diff(ib)**2))))
R = pd.DataFrame(rows); R.to_csv(f'{S}/verify/staircase.csv', index=False)
print(R.groupby('pattern').size().to_dict(), "recordings by sampling pattern")
print("long holds (>= 9 samples, i.e. dropped 100-Hz samples) per recording: median %d, max %d; time in long holds median %.1f%%, max %.1f%%" % (R.long_holds.median(), R.long_holds.max(), R.pct_time_in_long_holds.median(), R.pct_time_in_long_holds.max()))
print("beat-time shift staircase vs reconstructed: SD median %.2f ms (max %.2f); same-beat-count recordings IBI diff SD median %.2f ms" % (R.beat_shift_sd.median(), R.beat_shift_sd.max(), R.ibi_diff_sd.median()))
print("RMSSD staircase vs reconstructed: median %.1f vs %.1f ms" % (R.rmssd_stair.median(), R.rmssd_recon.median()))
