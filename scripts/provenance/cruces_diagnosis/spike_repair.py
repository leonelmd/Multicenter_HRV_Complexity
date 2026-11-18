import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ppg')); from ppg_lib import curve
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def spikes(x, k=5):
    d = np.diff(x); mad = 1.4826*np.median(np.abs(d - np.median(d))); return np.flatnonzero(np.abs(d - np.median(d)) > k*mad)
def repair(x, k=5):
    """Hampel-type: intervals adjacent to a spike difference replaced by the local median (11 beats)."""
    y = x.copy(); med = pd.Series(x).rolling(11, center=True, min_periods=1).median().values
    for i in spikes(x, k): y[i:i+2] = med[i:i+2]
    return y
ser = {}
for f in glob.glob(f'{S}/verify/out_rri/*.csv'): ser[('Cruces', os.path.basename(f))] = np.loadtxt(f)
for f in glob.glob(f'{S}/ppg/cetram/*/*_cleaned.csv'):
    x = np.diff(pd.read_csv(f)['sample'].values.astype(float)); x = x[(x >= 300) & (x <= 2000)]; ser[('CETRAM 500', os.path.basename(f))] = x[:500]
rows = []
for (c, s), x in ser.items():
    a, b = curve(x), curve(repair(x)); rows.append(dict(c=c, n_spk=len(spikes(x)), t1=a[0], t5=a[4], t1r=b[0], t5r=b[4]))
R = pd.DataFrame(rows)
for c, g in R.groupby('c'):
    print(f"{c:10s} spike pairs/recording median {g.n_spk.median():.0f} (IQR {g.n_spk.quantile(.25):.0f}-{g.n_spk.quantile(.75):.0f}) | "
          f"falling 1->5: {100*(g.t5 < g.t1).mean():.0f}% -> after repair {100*(g.t5r < g.t1r).mean():.0f}% | median t1 {g.t1.median():.2f} -> {g.t1r.median():.2f}")
