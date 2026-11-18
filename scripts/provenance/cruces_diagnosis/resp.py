import os, sys, numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from surrogates import iaaft, ser, curve
rows = []
for (c, s), x in ser.items():
    y = x - pd.Series(x).rolling(31, center=True, min_periods=1).mean().values
    f, p = welch(y, fs=1.0, nperseg=64); band = (f >= 0.1) & (f <= 0.4)
    peak = p[band].max()/p[band].sum()                 # sharpness of the respiratory peak (1 = single line)
    resp_share = p[band].sum()/p[f > 0.02].sum()
    a = curve(x); b = np.mean([curve(iaaft(x)) for _ in range(3)], axis=0)
    rows.append(dict(c=c, gap5=b[4]-a[4], peak=peak, resp_share=resp_share, falling=a[4] < a[0]))
R = pd.DataFrame(rows)
for c, g in R.groupby('c'):
    print(f"{c:10s} resp-band share {g.resp_share.median():.2f}, peak sharpness {g.peak.median():.2f} | rho(peak sharpness, surrogate gap at t5) {spearmanr(g.peak, g.gap5)[0]:+.2f} (p={spearmanr(g.peak, g.gap5)[1]:.3f}) | rho(resp share, gap) {spearmanr(g.resp_share, g.gap5)[0]:+.2f}")
g = R[R.c == 'Cruces']; print("Cruces falling curves: resp-band share %.2f vs rising %.2f" % (g[g.falling].resp_share.median(), g[~g.falling].resp_share.median()))
