#!/usr/bin/env python3
"""Split-half reliability of the complexity index as it is defined (Methods): the index is
computed separately on the first and second half of each analysed series, each half with
its own tolerance r = 0.2 SD(half), and reliability is the Spearman correlation of the two
across participants. Cruces (whole record, scales 1-5) and Nagoya (18:00-22:00, scales
1-20); for CETRAM the length ladder gives the same quantity (data/length_ladder.csv, N=1000).
Writes data/split_half_reliability.csv.
Needs the original Cruces and Nagoya series (paths.py).
"""
import os, glob, importlib.util
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rcmse_fast import rcmse
HERE = os.path.dirname(os.path.abspath(__file__))
from paths import DATA as D, CRUCES_RRI
def nauc(c): return float(np.trapezoid(c, np.arange(1, len(c) + 1)) / len(c)) if np.all(np.isfinite(c)) else np.nan
rows = []
grp = pd.read_csv(os.path.join(D, "spain_mse.csv")).drop_duplicates("Subject").set_index("Subject").Group
for f in sorted(glob.glob(os.path.join(CRUCES_RRI, "*.csv"))):
    s = os.path.basename(f)[:-4]
    if grp.get(s) not in ("Control", "PD"): continue
    x = np.loadtxt(f).astype(float); x = x[np.isfinite(x)]; h = len(x) // 2
    rows.append(dict(Cohort="Cruces", Subject=s, h1=nauc(rcmse(x[:h], 5)), h2=nauc(rcmse(x[h:2*h], 5)), N=len(x)))
spec = importlib.util.spec_from_file_location("w", os.path.join(HERE, "compute_japan_window_features.py"))
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
meta = pd.read_csv(w.NAGOYA_META_CSV).set_index("Subject_ID")
for s in pd.read_csv(os.path.join(D, "japan_window_features.csv")).Subject:
    x = w.extract_window_rri(os.path.join(w.NAGOYA_RRI_DIR, s + "_RRi.txt"), str(meta.loc[s, "Start_Time"])); h = len(x) // 2
    rows.append(dict(Cohort="Nagoya", Subject=s, h1=nauc(rcmse(x[:h], 20)), h2=nauc(rcmse(x[h:2*h], 20)), N=len(x)))
L = pd.read_csv(os.path.join(D, "length_ladder.csv")); L = L[(L.Cohort == "CETRAM") & (L.N == 1000)]
rows += [dict(Cohort="CETRAM (1000 beats)", Subject=r.Subject, h1=r.h1, h2=r.h2, N=1000) for r in L.itertuples()]
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "split_half_reliability.csv"), index=False)
for c, g in df.groupby("Cohort"):
    g = g.dropna(); print(c, len(g), "rho = %.2f" % spearmanr(g.h1, g.h2)[0])
