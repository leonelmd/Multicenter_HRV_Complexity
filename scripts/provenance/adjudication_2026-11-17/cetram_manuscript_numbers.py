"""Every CETRAM-dependent number quoted in the manuscript, recomputed from the released data.

Run from public_release/:  python3 scratch/adjudication_2026-11-17/cetram_manuscript_numbers.py
Written for the 2026-11-17 adjudicated exclusion (P004, C050, P012, P023). Run on the
pre-exclusion data it reproduces the numbers of the 2026-11-01 manuscript, which is how each
computation below was checked against its source.
"""
import os, sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = os.getcwd(); D = os.path.join(ROOT, "data"); SC = os.path.join(ROOT, "scratch")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = []


def say(k, v):
    OUT.append((k, v)); print(f"{k:58s} {v}")


def auc_p(c, p):
    c, p = np.asarray(c, float), np.asarray(p, float); c, p = c[~np.isnan(c)], p[~np.isnan(p)]
    u = stats.mannwhitneyu(c, p, alternative="two-sided")
    return u.statistic / len(c) / len(p), u.pvalue


def boot_ci(c, p, n=4000, seed=0):
    rng = np.random.default_rng(seed); c, p = np.asarray(c), np.asarray(p); a = []
    for _ in range(n):
        a.append(auc_p(rng.choice(c, len(c)), rng.choice(p, len(p)))[0])
    return np.percentile(a, [2.5, 97.5])


def nauc_tab(path, lo=1, hi=5):
    d = pd.read_csv(path); d = d[(d.Scales >= lo) & (d.Scales <= hi)].sort_values("Scales")
    return d.groupby(["Subject", "Group"]).MSE.apply(lambda c: np.trapezoid(c.values) / (hi - lo + 1)).rename("idx").reset_index()


cx = nauc_tab(os.path.join(D, "chile_mse.csv"))
met = pd.read_csv(os.path.join(D, "chile_metrics.csv")); met["HR"] = 60000 / met.HRV_MeanNN
dem = pd.read_csv(os.path.join(D, "chile_demographics.csv")).rename(columns={"Anon_ID": "Subject"})
x = cx.merge(met[["Subject", "HR", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_MeanNN"]], on="Subject").merge(dem[["Subject", "Age"]], on="Subject", how="left")
C, P = x[x.Group == "Control"], x[x.Group == "PD"]

print("== 1. sample")
say("n Control / PD / total", f"{len(C)} / {len(P)} / {len(x)}")
say("age C mean±SD | PD mean±SD", f"{C.Age.mean():.1f}±{C.Age.std():.1f} | {P.Age.mean():.1f}±{P.Age.std():.1f}")
say("age missing (n)", int(x.Age.isna().sum()))
beats = []
for g in ["Control", "PD"]:
    det = os.path.join(ROOT, "..", "..", "CETRAM", "public_release", "results", "cleaned_detections", g)
    for s in x[x.Group == g].Subject:
        f = os.path.join(det, f"{s}_cleaned.csv")
        if os.path.exists(f):
            beats.append(len(pd.read_csv(f)) - 1)
if beats:
    say("mean intervals per recording", f"{np.mean(beats):.0f} (n={len(beats)})")

print("== 2. conventional metrics (AUC: >0.5 lower in PD; HR oriented higher in PD)")
for col, lab in [("idx", "complexity"), ("HRV_SDNN", "SDNN"), ("HRV_RMSSD", "RMSSD"), ("HRV_pNN50", "pNN50")]:
    a, p = auc_p(C[col], P[col]); say(f"{lab} AUC (p)", f"{a:.3f} ({p:.3f})")
a, p = auc_p(P.HR, C.HR); say("HR AUC (p)", f"{a:.3f} ({p:.3f})")
say("SDNN median C vs PD", f"{C.HRV_SDNN.median():.1f} vs {P.HRV_SDNN.median():.1f}")
say("RMSSD median C vs PD", f"{C.HRV_RMSSD.median():.1f} vs {P.HRV_RMSSD.median():.1f}")

print("== 3. primary")
a, p = auc_p(C.idx, P.idx); lo, hi = boot_ci(C.idx.values, P.idx.values)
say("median [IQR] C", f"{C.idx.median():.3f} [{C.idx.quantile(.25):.3f}--{C.idx.quantile(.75):.3f}]")
say("median [IQR] PD", f"{P.idx.median():.3f} [{P.idx.quantile(.25):.3f}--{P.idx.quantile(.75):.3f}]")
say("AUC [95% CI] p", f"{a:.3f} [{lo:.3f}--{hi:.3f}] p={p:.4f}")
d = x.copy(); d["g"] = (d.Group == "PD").astype(int)
m = sm.OLS(d.idx, sm.add_constant(d[["g", "HR"]])).fit(cov_type="HC3")
adj = roc_auc_score(d.g, -(d.idx - m.params["HR"] * d.HR))
dz = d.copy(); dz["idx"] = (dz.idx - dz.idx.mean()) / dz.idx.std(); dz["HR"] = (dz.HR - dz.HR.mean()) / dz.HR.std()
mz = sm.OLS(dz.idx, sm.add_constant(dz[["g", "HR"]])).fit(cov_type="HC3")
say("HR-adjusted p | adj AUC | std beta (p)", f"{m.pvalues['g']:.4f} | {adj:.3f} | {mz.params['g']:.3f} ({mz.pvalues['g']:.4f})")

print("== 4. age (Spearman within group; scale-mean index as in Fig. 5, and nAUC)")
mse = pd.read_csv(os.path.join(D, "chile_mse.csv"))
sm_ = mse[mse.Scales.between(1, 5)].groupby("Subject").MSE.mean().rename("smean").reset_index()
xa = x.merge(sm_, on="Subject")
for g in ["Control", "PD"]:
    s = xa[xa.Group == g].dropna(subset=["Age"])
    r1 = stats.spearmanr(s.Age, s.smean); r2 = stats.spearmanr(s.Age, s.idx)
    say(f"age rho {g} (scale-mean | nAUC)", f"{r1[0]:+.2f} p={r1[1]:.2f} | {r2[0]:+.2f} p={r2[1]:.2f}  n={len(s)}")

print("== 5. SDNN>100 ms set and sensitivity")
hi100 = x[x.HRV_SDNN > 100]
say("SDNN>100 ms recordings", f"{sorted(hi100.Subject)} ({(hi100.Group=='PD').sum()} PD)")
y = x[x.HRV_SDNN <= 100]
a, p = auc_p(y[y.Group == "Control"].idx, y[y.Group == "PD"].idx); say("complexity excl SDNN>100", f"AUC {a:.3f} p={p:.3f} n={len(y)}")
for col in ["HRV_SDNN", "HRV_RMSSD"]:
    a, p = auc_p(y[y.Group == "Control"][col], y[y.Group == "PD"][col]); say(f"{col} excl SDNN>100", f"AUC {a:.3f} p={p:.3f}")
sd = pd.read_csv(os.path.join(D, "sdnn_min_all_centers.csv")); sd = sd[sd.Center == "Chile"]
a, p = auc_p(sd[sd.Group == "Control"].SDNN100_min, sd[sd.Group == "PD"].SDNN100_min); say("min SDNN all", f"AUC {a:.3f} p={p:.3f} n={len(sd)}")
sy = sd[sd.Subject.isin(y.Subject)]
a, p = auc_p(sy[sy.Group == "Control"].SDNN100_min, sy[sy.Group == "PD"].SDNN100_min); say("min SDNN excl SDNN>100", f"AUC {a:.3f}")
r = x.idx.values; tol = 0.2 * x.HRV_SDNN.values
say("tolerance r (ms) in SDNN>100 set", f"{0.2*hi100.HRV_SDNN.min():.0f}--{0.2*hi100.HRV_SDNN.max():.0f}")

print("== 6. irregular-rhythm flag (hf_frac of scratch/standardized/std_series.py, on the analysed series)")
from scipy.signal import welch
def hf_frac(v):
    yv = v - pd.Series(v).rolling(31, center=True, min_periods=1).mean().values
    f, pw = welch(yv, fs=1.0, nperseg=min(128, len(yv) // 2)); return pw[f >= 0.35].sum() / pw[f > 0.02].sum()
DET = os.path.join(ROOT, "..", "..", "CETRAM", "public_release", "results", "cleaned_detections")
RR = {s_: np.diff(pd.read_csv(os.path.join(DET, g_, f"{s_}_cleaned.csv"))["sample"].values.astype(float)) for s_, g_ in zip(x.Subject, x.Group)}
hf = pd.Series({s_: hf_frac(v) for s_, v in RR.items()})
fl = hf[hf >= 0.30].index
say("flagged CETRAM", f"{sorted(fl)} ({sum(s_.startswith('P') for s_ in fl)} PD)")
y = x[~x.Subject.isin(fl)]; a, p = auc_p(y[y.Group == "Control"].idx, y[y.Group == "PD"].idx)
say("complexity excl flagged", f"AUC {a:.3f} p={p:.3f}")
say("median alternating-band share", f"{hf.median():.3f}")

print("== 7. Pan-Tompkins vs Brammer")
pt = pd.read_csv(os.path.join(SC, "rerun_2026-11-01", "pt_vs_brammer.csv")); pt = pt[pt.Subject.isin(x.Subject)]
for det in ["pantompkins1985", "brammer"]:
    q = pt[pt.det == det] if det in set(pt.det) else pt[pt.det != "brammer"]
    a, p = auc_p(q[q.Group == "Control"].idx, q[q.Group == "PD"].idx)
    cm = q[q.Group == "Control"][["s1", "s5"]].mean(); pm = q[q.Group == "PD"][["s1", "s5"]].mean()
    say(f"{det} AUC p | C s1->s5 | PD s1->s5", f"{a:.3f} p={p:.3f} | {cm.s1:.2f}->{cm.s5:.2f} | {pm.s1:.2f}->{pm.s5:.2f}  (dets in file: {sorted(set(pt.det))})")

print("== 8. parameter sweep (r_sweep.csv)")
rs = pd.read_csv(os.path.join(D, "r_sweep.csv")); rc = rs[(rs.Cohort == "CETRAM")]
def grid_auc(g):
    c, p = g[g.Group == "Control"].cx, g[g.Group == "PD"].cx
    return auc_p(c, p)[0]
k = rc[rc["mode"] == "k"]
usable = k[(k.level >= 0.125)]
aucs = usable.groupby(["m", "level"]).apply(grid_auc)
say("AUC range usable grid (max-min)", f"{aucs.max()-aucs.min():.3f}  [{aucs.min():.3f}--{aucs.max():.3f}]")
und = k[(k.level == 0.05) & (k.m == 3)]
say("undefined at k=0.05 m=3", f"{100*und.cx.isna().mean():.1f}%")
m2 = k[k.m == 2].groupby("level").apply(grid_auc)
say("optimum k at m=2 | gain over k=0.2", f"{m2.idxmax()} | {m2.max()-m2.loc[0.2]:.3f}")
ab = rc[rc["mode"] == "abs"]
aa = ab.groupby(["m", "level"]).apply(grid_auc)
say("abs tolerance AUC range (m=2, 4-60 ms)", f"{aa.loc[2].min():.2f}--{aa.loc[2].max():.2f}")
rr = ab[ab.m == 2].groupby("level").apply(lambda g: stats.spearmanr(g.r_ms if g.r_ms.nunique() > 1 else g.SD, g.SD)[0])
def sh(g):
    g = g.dropna(subset=["h1", "h2"]); return stats.spearmanr(g.h1, g.h2)[0] if len(g) > 3 else np.nan
say("split-half reliability k=0.2 m=2 (Spearman h1,h2)", f"{sh(k[(k.m==2)&(np.isclose(k.level,0.2))]):.2f}")
say("split-half reliability k=0.2 m=2 (Pearson h1,h2)", f"{k[(k.m==2)&(np.isclose(k.level,0.2))][['h1','h2']].dropna().corr().iloc[0,1]:.2f}")

print("== 9. length ladder (CETRAM, common subset)")
ll = pd.read_csv(os.path.join(D, "length_ladder.csv")); lc = ll[ll.Cohort == "CETRAM"]
for n in [250, 1000]:
    g = lc[lc.N == n]
    say(f"ladder AUC at {n}", f"{auc_p(g[g.Group=='Control'].nAUC, g[g.Group=='PD'].nAUC)[0]:.3f} (n={len(g)})")

print("== 10. Cruces-section CETRAM comparators")
sv = pd.read_csv(os.path.join(SC, "cruces_diagnosis", "shape_vs_spectrum.csv"))
ce = sv[sv.cohort.str.startswith("CETRAM")].copy(); ce["Subject"] = ce.s.str[:4]; ce = ce[ce.Subject.isin(x.Subject)]
say("CETRAM 500-beat curves falling t1->t5", f"{100*(ce.t5 < ce.t1).mean():.0f}% (n={len(ce)})")

sys.path.insert(0, os.path.join(SC, "standardized")); from ppg_lib import curve
rng = np.random.default_rng(0)
def iaaft(v, it=50):
    amp = np.abs(np.fft.rfft(v - v.mean())); srt = np.sort(v); yv = rng.permutation(v)
    for _ in range(it):
        Y = np.fft.rfft(yv - yv.mean()); yv = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=len(v)) + v.mean()
        yv = srt[np.argsort(np.argsort(yv))]
    return yv
fall_r, fall_s, resp = [], [], []
for s_, v in RR.items():
    v = v[(v >= 300) & (v <= 2000)][:500]
    a_ = curve(v); b_ = np.mean([curve(iaaft(v)) for _ in range(3)], axis=0)
    fall_r.append(a_[4] < a_[0]); fall_s.append(b_[4] < b_[0])
    yv = v - pd.Series(v).rolling(31, center=True, min_periods=1).mean().values
    f, pw = welch(yv, fs=1.0, nperseg=64); band = (f >= 0.1) & (f <= 0.4); resp.append(pw[band].sum() / pw[f > 0.02].sum())
say("CETRAM 500: falling real | surrogate | resp-band share", f"{100*np.mean(fall_r):.0f}% | {100*np.mean(fall_s):.0f}% | {np.median(resp):.2f}")

print("== 11. per-scale AUC")
for t in range(1, 6):
    s = mse[mse.Scales == t]; say(f"per-scale AUC tau={t}", f"{auc_p(s[s.Group=='Control'].MSE, s[s.Group=='PD'].MSE)[0]:.3f}")

print("== 12. clinical (generate_figure8 machinery)")
import generate_figure8 as g8
df, nm_cols = g8.load_data()
say("PD with clinical record", f"{len(df)}")
say("LEDD available", f"{df['LEDD'].notna().sum()}")
CONT = ['Disease_duration', 'HY_stage', 'Schwab_England', 'CISI_motor', 'CISI_disability', 'CISI_motor_comp', 'CISI_cognitive', 'CISI_total', 'NM_burden']
crude, adjr = [], []
for v in CONT:
    s = df[["Complexity", v]].dropna(); r = stats.spearmanr(s.Complexity, s[v]); crude.append((v, r[0], r[1]))
    np.random.seed(0); ra = g8.partial_spearman_age(df, "Complexity", v, cov_col="Age"); adjr.append((v, ra[0], ra[1]))
qc = g8.bh_fdr(np.array([c[2] for c in crude])); qa = g8.bh_fdr(np.array([c[2] for c in adjr]))
say("min q crude | min q age-adjusted", f"{np.nanmin(qc):.2f} | {np.nanmin(qa):.2f}")
i = int(np.nanargmin([c[2] for c in adjr])); say("strongest age-adjusted", f"{adjr[i][0]} rho={adjr[i][1]:+.2f} p={adjr[i][2]:.3f} q={qa[i]:.2f}")
cl = pd.read_excel(os.path.join(D, "deidentified_clinical_consolidated.xlsx"))
an = cl[cl.Anonymous_ID.isin(P.Subject)]
say("cardiac-history blank among analysed PD", f"{an['Cardiac problems'].isna().sum()} of {len(P)}")
s = df[["LEDD", "Complexity"]].dropna(); r = stats.spearmanr(s.LEDD, s.Complexity)
say("LEDD vs complexity Spearman", f"rho={r[0]:+.2f} p={r[1]:.2f} n={len(s)}")

print("== 13. orthostatic hypotension (exploratory)")
oh = cl[["Anonymous_ID", "Orthostatic hypotension", "Orthostatic hypotension (Estimated)", "Autonomic disorder clinical index", "Age"]].rename(columns={"Anonymous_ID": "Subject"})
o = P.merge(oh, on="Subject", suffixes=("", "_cl"))
o = o.dropna(subset=["Orthostatic hypotension"])
w, wo = o[o["Orthostatic hypotension"] == 1], o[o["Orthostatic hypotension"] == 0]
a, p = auc_p(wo.idx, w.idx)
say("OH available | with | without", f"{len(o)} | {len(w)} | {len(wo)}")
say("OH median with vs without | AUC p", f"{w.idx.median():.2f} vs {wo.idx.median():.2f} | {a:.3f} p={p:.3f}")
y = o[~o.Subject.isin(hi100.Subject)]
a, p = auc_p(y[y["Orthostatic hypotension"] == 0].idx, y[y["Orthostatic hypotension"] == 1].idx)
say("OH excl SDNN>100 PD", f"AUC {a:.3f} p={p:.3f} (dropped {sorted(set(o.Subject) & set(hi100.Subject))})")
est = o["Orthostatic hypotension (Estimated)"].astype(str).str.lower().isin(["true", "1", "1.0"])
y = o[~est]; a, p = auc_p(y[y["Orthostatic hypotension"] == 0].idx, y[y["Orthostatic hypotension"] == 1].idx)
say("OH excl imputed entry", f"AUC {a:.3f} p={p:.3f} (imputed: {sorted(o[est].Subject)})")
age_col = "Age" if o["Age"].notna().sum() >= o["Age_cl"].notna().sum() else "Age_cl"
z = o.dropna(subset=["Age_cl", "HR"]).copy(); z["rk"] = z.idx.rank()
mm = sm.OLS(z.rk, sm.add_constant(z[["Orthostatic hypotension", "Age_cl", "HR"]].astype(float))).fit(cov_type="HC3")
say("OH rank regression (age, HR; HC3) p", f"{mm.pvalues['Orthostatic hypotension']:.3f} (n={len(z)})")
ai = P.merge(oh[["Subject", "Autonomic disorder clinical index"]], on="Subject").dropna(subset=["Autonomic disorder clinical index"])
r = stats.spearmanr(ai.idx, ai["Autonomic disorder clinical index"])
say("autonomic clinical index", f"n={len(ai)} rho={r[0]:+.2f} p={r[1]:.2f}")

pd.DataFrame(OUT, columns=["quantity", "value"]).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"numbers_{sys.argv[1] if len(sys.argv) > 1 else 'run'}.csv"), index=False)
