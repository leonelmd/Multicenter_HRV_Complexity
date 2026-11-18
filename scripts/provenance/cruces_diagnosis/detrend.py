import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from surrogates import iaaft, ser, curve
rows = []
for (c, s), x in ser.items():
    y = x - pd.Series(x).rolling(61, center=True, min_periods=1).mean().values + x.mean()   # remove trends slower than ~60 beats
    a = curve(y); b = np.mean([curve(iaaft(y)) for _ in range(3)], axis=0)
    h = len(x)//2; step = abs(x[:h].mean() - x[h:].mean())/np.std(x, ddof=1)
    q = np.array_split(x, 4); drift = np.std([v.mean() for v in q])/np.std(x, ddof=1)
    rows.append(dict(c=c, t1=a[0], t5=a[4], t1s=b[0], t5s=b[4], drift=drift))
R = pd.DataFrame(rows)
for c, g in R.groupby('c'):
    print(f"{c:10s} after removing >60-beat trends: falling original {100*(g.t5 < g.t1).mean():.0f}% vs surrogates {100*(g.t5s < g.t1s).mean():.0f}% | t5 {g.t5.median():.2f} vs {g.t5s.median():.2f} || drift between quarters (SD of quarter means / SD): {g.drift.median():.2f}")
