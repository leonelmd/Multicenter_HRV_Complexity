#!/usr/bin/env python3
"""
Artefact sensitivity of the complexity index (response to the Supplementary Fig. S8B concern).

Per recording, on the analysed series (loaders of compute_spectral_analysis.series):
  rate_LT     share of beats flagged by Lipponen-Tarvainen (NeuroKit2 signal_fixpeaks,
              Kubios, iterative): ectopic + missed + extra + long/short
  rate_dev20  share of intervals deviating > 20 % from the 31-beat running median
  cx          primary index (nAUC 1-5; Nagoya 1-20), recomputed here
  cx_LT       index after Lipponen-Tarvainen correction of the series
  cx_mad      index with r = 0.2 x 1.4826 x MAD (robust to outlying intervals)
  cx_drop     index after deleting intervals flagged by the 20 % rule
  cx_interp   index after replacing those intervals by the 31-beat running median
              (--interp; written to data/artefact_interp.csv)
Output: data/artefact_sensitivity.csv (one row per recording), resumable by part files.
Then: python scripts/compute_artefact_sensitivity.py --report

Missed-beat injection (realistic artefact, passes the 300-2000 ms guard): pairs of
consecutive intervals merged at 0.1-1 % of beats, 12 recordings per cohort,
-> data/artefact_injection.csv
"""
import os, sys, time, zlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_spectral_analysis import series
from rcmse_fast import rcmse
import compute_length_ladder as L
from scipy.stats import mannwhitneyu, spearmanr

D = L.DATA
PARTS = os.path.join(D, "_artefact_parts")
NS = {"CETRAM": 5, "Cruces": 5, "Nagoya": 20}


def nauc(c):
    c = np.asarray(c, float)
    return float(np.trapezoid(c, np.arange(1, len(c) + 1)) / len(c)) if np.isfinite(c).all() else np.nan


def idx(x, ns, r=None, k=0.2):
    if r is None:
        r = k * x.std(ddof=1)
    return nauc(rcmse_r(x, ns, r))


def rcmse_r(x, ns, r, m=2):
    from rcmse_fast import _AB
    N = len(x); out = []
    for t in range(1, ns + 1):
        A = B = 0.0
        for o in range(t):
            Lh = (N - o) // t
            if Lh <= m: continue
            a, b = _AB(x[o:o + Lh * t].reshape(Lh, t).mean(1), m, r); A += a; B += b
        out.append(-np.log(A / B) if A > 0 and B > 0 else np.nan)
    return np.array(out)


def lt_correct(x):
    import neurokit2 as nk
    t = np.concatenate([[0.0], np.cumsum(x)])
    pk = np.round(t * 10).astype(np.int64)            # 0.1-ms resolution
    info, pk2 = nk.signal_fixpeaks(pk, sampling_rate=10000, method="Kubios", iterative=True, show=False)
    n = sum(len(info[k]) for k in ("ectopic", "missed", "extra", "longshort"))
    y = np.diff(np.asarray(pk2, float)) / 10.0
    return y[(y >= 300) & (y <= 2000)], n


def dev20(x):
    med = pd.Series(x).rolling(31, center=True, min_periods=5).median().values
    return np.abs(x - med) > 0.2 * med


def one(coh, s, g, x):
    ns = NS[coh]
    y, nlt = lt_correct(x)
    bad = dev20(x)
    mad = 1.4826 * np.median(np.abs(x - np.median(x)))
    return dict(Cohort=coh, Subject=s, Group=g, n=len(x), rate_LT=nlt / len(x), rate_dev20=bad.mean(),
                cx=idx(x, ns), cx_LT=idx(y, ns), cx_mad=idx(x, ns, r=0.2 * mad),
                cx_drop=idx(x[~bad], ns))


def inject(x, rate, rng):
    x = x.copy(); n = int(round(rate * len(x)))
    if n == 0: return x
    pos = np.sort(rng.choice(np.arange(1, len(x) - 1), n, replace=False))
    keep = np.ones(len(x), bool)
    for p in pos:
        if keep[p] and keep[p - 1]:
            x[p - 1] = x[p - 1] + x[p]; keep[p] = False      # missed beat: two intervals merge
    x = x[keep]
    return x[(x >= 300) & (x <= 2000)]


def run(tmax):
    os.makedirs(PARTS, exist_ok=True); t0 = time.time()
    for coh, s, g, x in series():
        f = os.path.join(PARTS, f"{coh}_{s}.csv")
        if os.path.exists(f): continue
        if time.time() - t0 > tmax: print("PAUSE"); return False
        pd.DataFrame([one(coh, s, g, x)]).to_csv(f, index=False)
    inj = os.path.join(D, "artefact_injection.csv"); ip = os.path.join(D, "_artefact_inj")
    os.makedirs(ip, exist_ok=True)
    for coh, s, g, x in series():
        if coh == "Cruces" or zlib.crc32(s.encode()) % 4:   # ~1 in 4 recordings, label-blind
            continue
        f = os.path.join(ip, f"{coh}_{s}.csv")
        if os.path.exists(f): continue
        if time.time() - t0 > tmax: print("PAUSE (injection)"); return False
        ns = NS[coh]; base = idx(x, ns); rng = np.random.default_rng(zlib.crc32(s.encode()))
        pd.DataFrame([dict(Cohort=coh, Subject=s, rate=rate, rel=idx(inject(x, rate, rng), ns) / base)
                      for rate in (0.001, 0.002, 0.005, 0.01)]).to_csv(f, index=False)
    import glob
    pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ip, "*.csv"))]).to_csv(inj, index=False)
    return True


def report():
    import glob
    A = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(PARTS, "*.csv"))])
    A.to_csv(os.path.join(D, "artefact_sensitivity.csv"), index=False)
    def auc(df, col):
        a, b = df[df.Group == "Control"][col].dropna(), df[df.Group == "PD"][col].dropna()
        u = mannwhitneyu(a, b, alternative="two-sided"); return u.statistic / (len(a) * len(b)), u.pvalue
    for coh, d in A.groupby("Cohort"):
        print(f"\n== {coh}  n={len(d)}")
        for col in ("rate_LT", "rate_dev20"):
            c, p = d[d.Group == "Control"][col], d[d.Group == "PD"][col]
            print(f"  {col:10s} median C {100*c.median():.3f}%  PD {100*p.median():.3f}%  "
                  f"MWU p={mannwhitneyu(c, p).pvalue:.3f}   rho(cx) C {spearmanr(c, d[d.Group=='Control'].cx)[0]:+.2f} "
                  f"PD {spearmanr(p, d[d.Group=='PD'].cx)[0]:+.2f}")
        for col in ("cx", "cx_LT", "cx_mad", "cx_drop"):
            a, pv = auc(d, col); print(f"  AUC {col:8s} {a:.3f}  p={pv:.4f}")
        import statsmodels.api as sm
        z = d.dropna(subset=["cx"]).copy(); z["g"] = (z.Group == "PD").astype(float)
        for col in ("rate_LT", "rate_dev20"):
            m = sm.OLS((z.cx - z.cx.mean()) / z.cx.std(), sm.add_constant(z[["g", col]])).fit(cov_type="HC3")
            print(f"  cx ~ group + {col}: beta_PD {m.params['g']:+.3f} p={m.pvalues['g']:.4f}")
    inj = os.path.join(D, "artefact_injection.csv")
    if os.path.exists(inj):
        I = pd.read_csv(inj)
        print("\nmissed-beat injection, index relative to clean (median [IQR]):")
        print(I.groupby(["Cohort", "rate"]).rel.describe()[["count", "25%", "50%", "75%"]].round(3))


def interp():
    rows = []
    for coh, s, g, x in series():
        bad = dev20(x); med = pd.Series(x).rolling(31, center=True, min_periods=5).median().values
        y = x.copy(); y[bad] = med[bad]
        rows.append(dict(Cohort=coh, Subject=s, Group=g, cx_interp=idx(y, NS[coh])))
    pd.DataFrame(rows).to_csv(os.path.join(D, "artefact_interp.csv"), index=False)


if __name__ == "__main__":
    if "--interp" in sys.argv:
        interp()
    elif "--report" in sys.argv:
        report()
    else:
        tm = float(sys.argv[sys.argv.index("--max-seconds") + 1]) if "--max-seconds" in sys.argv else 1e12
        print("DONE" if run(tm) else "")
