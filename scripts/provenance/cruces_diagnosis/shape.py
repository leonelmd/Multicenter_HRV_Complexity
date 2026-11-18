import os, numpy as np, pandas as pd, glob
from scipy.signal import welch
from scipy.stats import spearmanr
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = pd.read_csv(f'{S}/verify/curves20.csv')
ser = {}
for f in glob.glob(f'{S}/verify/out_rri/*.csv'): ser[('Cruces', os.path.basename(f)[:-4])] = np.loadtxt(f)
for f in glob.glob(f'{S}/ppg/cetram/*/*_cleaned.csv'):
    x = np.diff(pd.read_csv(f)['sample'].values.astype(float)); x = x[(x >= 300) & (x <= 2000)]; ser[('CETRAM 500 beats', os.path.basename(f))] = x[:500]
for f in sorted(glob.glob(f'{S}/dur/data/processed_rri/*_RRi.txt')):
    d = np.loadtxt(f); x = d[:, 1]*1000; x = x[(x >= 300) & (x <= 2000)]; ser[('Nagoya 500 beats', os.path.basename(f))] = x[20000:20500] if len(x) > 20500 else x[:500]
rows = []
for (c, s), x in ser.items():
    y = x - x.mean(); f, p = welch(y, fs=1.0, nperseg=128); m = (f > 0.01)
    beta = -np.polyfit(np.log(f[m]), np.log(p[m]), 1)[0]          # spectral exponent, 1/f^beta, beat domain
    slow = p[(f > 0.01) & (f < 0.1)].sum()/p[m].sum()
    rows.append(dict(cohort=c, s=s, beta=beta, slow=slow))
B = pd.DataFrame(rows).merge(C, on=['cohort', 's']); B['slope15'] = B.t5 - B.t1
print("spectral exponent beta (median): " + "  ".join(f"{c} {g.beta.median():.2f}" for c, g in B.groupby('cohort')))
print("Spearman rho(beta, t5 - t1) within each cohort: " + "  ".join(f"{c} {spearmanr(g.beta, g.slope15)[0]:+.2f}" for c, g in B.groupby('cohort')))
print("pooled across cohorts: rho(beta, slope) = %+.2f" % spearmanr(B.beta, B.slope15)[0])
# does cohort add anything once beta is known? simple OLS slope ~ beta + cohort dummies
X = pd.get_dummies(B[['beta', 'cohort']], columns=['cohort'], drop_first=True).astype(float); X.insert(0, 'c', 1.0)
b, *_ = np.linalg.lstsq(X.values, B.slope15.values, rcond=None); print("OLS slope15 ~ beta + cohort:", dict(zip(X.columns, np.round(b, 3))))
# predicted share of falling curves for a cohort at its beta, from the ECG-only fit
E = B[B.cohort != 'Cruces']; be = np.polyfit(E.beta, E.slope15, 1)
pred = np.polyval(be, B[B.cohort == 'Cruces'].beta)
print("Cruces: observed falling 1->5 %.0f%%; predicted from its beta using the ECG relation %.0f%%" % (100*(B[B.cohort=='Cruces'].slope15 < 0).mean(), 100*(pred < 0).mean()))
B.to_csv(f'{S}/verify/shape_vs_spectrum.csv', index=False)
