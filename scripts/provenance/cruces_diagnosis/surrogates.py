import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ppg')); from ppg_lib import curve
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); rng = np.random.default_rng(0)
def iaaft(x, it=50):
    amp = np.abs(np.fft.rfft(x - x.mean())); srt = np.sort(x); y = rng.permutation(x)
    for _ in range(it):
        Y = np.fft.rfft(y - y.mean()); y = np.fft.irfft(amp*np.exp(1j*np.angle(Y)), n=len(x)) + x.mean()
        y = srt[np.argsort(np.argsort(y))]
    return y
ser = {}
for f in glob.glob(f'{S}/verify/out_rri/*.csv'): ser[('Cruces', os.path.basename(f))] = np.loadtxt(f)
for f in glob.glob(f'{S}/ppg/cetram/*/*_cleaned.csv'):
    x = np.diff(pd.read_csv(f)['sample'].values.astype(float)); x = x[(x >= 300) & (x <= 2000)]; ser[('CETRAM 500', os.path.basename(f))] = x[:500]
if __name__ == "__main__":
    rows = []
    if __name__ != "__main__": pass
    for (c, s), x in ser.items():
        a = curve(x); b = np.mean([curve(iaaft(x)) for _ in range(3)], axis=0)
        rows.append(dict(c=c, t1=a[0], t5=a[4], t1s=b[0], t5s=b[4]))
    R = pd.DataFrame(rows)
    for c, g in R.groupby('c'):
        print(f"{c:10s} falling 1->5: original {100*(g.t5 < g.t1).mean():.0f}%  vs surrogates {100*(g.t5s < g.t1s).mean():.0f}% | median t1 {g.t1.median():.2f} vs {g.t1s.median():.2f} | t5 {g.t5.median():.2f} vs {g.t5s.median():.2f}")
