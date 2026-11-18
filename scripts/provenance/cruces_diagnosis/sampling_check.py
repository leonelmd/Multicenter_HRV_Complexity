import glob, os, numpy as np
from scipy.signal import welch
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(f'{S}/fid/ppg_traces/*.csv'))
print("file   unique-values  median run of identical samples  share of samples where value changes  spectral power >25 Hz (fraction)")
runs_all = []
for f in files[:12]:
    x = np.loadtxt(f); ch = np.flatnonzero(np.diff(x) != 0); runs = np.diff(ch)
    fr, p = welch(x - x.mean(), fs=500, nperseg=4096)
    print(f"{os.path.basename(f):10s} {len(np.unique(x)):6d}   {np.median(runs):5.1f} (mode {np.bincount(runs).argmax()})   {len(ch)/len(x):.3f}   {p[fr > 25].sum()/p[fr > 0.3].sum():.2e}")
