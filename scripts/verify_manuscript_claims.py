#!/usr/bin/env python3
"""Recompute every headline number claimed in the manuscript from released data.

Claims updated 2026-11-17 for the CETRAM trace-review exclusions (P004, C050, P012, P023;
see CETRAM/public_release/data/metadata/adjudicated_exclusions.csv).

This does not read the summary CSVs where it can avoid them: AUCs and p values are
recomputed from the per-subject files so that a stale intermediate cannot hide a
mismatch. Each check prints CLAIM, COMPUTED and a verdict.

Run:  python3 scripts/verify_manuscript_claims.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")

FAILS, CHECKS = [], 0


def auc(control, pd_):
    """AUC for Control > PD, via the Mann-Whitney U identity."""
    c, p = np.asarray(control, float), np.asarray(pd_, float)
    c, p = c[~np.isnan(c)], p[~np.isnan(p)]
    u = stats.mannwhitneyu(c, p, alternative="two-sided")
    return u.statistic / (len(c) * len(p)), u.pvalue


def chk(label, claim, got, tol=0.0015, fmt="{:.3f}"):
    global CHECKS
    CHECKS += 1
    ok = got is not None and abs(claim - got) <= tol
    if not ok:
        FAILS.append(label)
    gs = "n/a" if got is None else fmt.format(got)
    print(f"  [{'OK ' if ok else 'XX '}] {label:<52} claim {fmt.format(claim):>9}   got {gs:>9}")


def nauc(df, lo, hi, scol="Scales", vcol="MSE"):
    """Canonical complexity index: trapz(curve, 1..n) / n  (see complexity_index.py)."""
    s = df[(df[scol] >= lo) & (df[scol] <= hi)].sort_values(scol)
    c = s[vcol].to_numpy(float)
    if len(c) == 0 or not np.isfinite(c).all():
        return np.nan
    return float(np.trapezoid(c, np.arange(1, len(c) + 1)) / len(c))


def index_by_subject(path, lo, hi, scol="Scales", vcol="MSE"):
    d = pd.read_csv(path)
    return (d.groupby(["Subject", "Group"])
             .apply(lambda g: nauc(g, lo, hi, scol, vcol), include_groups=False)
             .rename("idx").reset_index())


print("\n" + "=" * 96)
print("1. SAMPLE SIZES")
print("=" * 96)
ch = pd.read_csv(os.path.join(D, "chile_metrics.csv"))
sp = pd.read_csv(os.path.join(D, "spain_metrics.csv"))
ja = pd.read_csv(os.path.join(D, "japan_window_features.csv"))
for name, df, c_claim, p_claim in [("CETRAM", ch, 42, 27), ("Cruces", sp, 21, 31),
                                   ("Nagoya 18-22h", ja, 21, 26)]:
    vc = df.Group.value_counts()
    nc = int(vc.get("Control", 0)); npd = int(vc.get("PD", 0))
    chk(f"{name} n Control", c_claim, nc, tol=0, fmt="{:.0f}")
    chk(f"{name} n PD", p_claim, npd, tol=0, fmt="{:.0f}")

print("\n" + "=" * 96)
print("2. PRIMARY COMPLEXITY RESULTS")
print("=" * 96)
cet = index_by_subject(os.path.join(D, "chile_mse.csv"), 1, 5)
a, p = auc(cet[cet.Group == "Control"].idx, cet[cet.Group == "PD"].idx)
chk("CETRAM complexity AUC", 0.655, a, tol=0.004)
chk("CETRAM complexity p", 0.0309, p, tol=0.0005, fmt="{:.4f}")

cru = index_by_subject(os.path.join(D, "spain_mse.csv"), 1, 5)
a, p = auc(cru[cru.Group == "Control"].idx, cru[cru.Group == "PD"].idx)
chk("Cruces complexity AUC", 0.657, a, tol=0.004)
chk("Cruces complexity p", 0.057, p, tol=0.002, fmt="{:.3f}")

nag = index_by_subject(os.path.join(D, "japan_window_mse.csv"), 1, 20)
a20, p20 = auc(nag[nag.Group == "Control"].idx, nag[nag.Group == "PD"].idx)
chk("Nagoya 18-22h AUC, scales 1-20", 0.815, a20, tol=0.004)
chk("Nagoya 18-22h p < 0.001, scales 1-20", 1, float(p20 < 0.001), tol=0, fmt="{:.0f}")
nag5 = index_by_subject(os.path.join(D, "japan_window_mse.csv"), 1, 5)
a5, _ = auc(nag5[nag5.Group == "Control"].idx, nag5[nag5.Group == "PD"].idx)
chk("Nagoya 18-22h AUC, scales 1-5", 0.842, a5, tol=0.004)

print("\n" + "=" * 96)
print("3. SUZUKI BENCHMARK REPRODUCTION")
print("=" * 96)
sd = pd.read_csv(os.path.join(D, "sdnn_min_all_centers.csv"))
nj = sd[sd.Center.astype(str).str.lower().str.contains("japan|nagoya")]
nj = nj[nj.Subject.isin(set(pd.read_csv(os.path.join(D, "japan_window_features.csv")).Subject))]
if len(nj) == 0:
    print(f"  !! no Nagoya rows; centers present: {sorted(sd.Center.unique())}")
else:
    njc = nj[nj.Group != "PD"].SDNN100_min
    njp = nj[nj.Group == "PD"].SDNN100_min
    chk("Nagoya min-SDNN100 control mean", 10.63, njc.mean(), tol=0.05, fmt="{:.2f}")
    chk("Nagoya min-SDNN100 control SD", 4.30, njc.std(ddof=1), tol=0.05, fmt="{:.2f}")
    chk("Nagoya min-SDNN100 PD mean", 4.85, njp.mean(), tol=0.05, fmt="{:.2f}")
    chk("Nagoya min-SDNN100 PD SD", 2.17, njp.std(ddof=1), tol=0.05, fmt="{:.2f}")
    a, _ = auc(njc, njp)
    chk("Nagoya min-SDNN100 AUC", 0.905, a, tol=0.004)
    cc, cp = nj[nj.Group != "PD"].CVRR100_min, nj[nj.Group == "PD"].CVRR100_min
    chk("Nagoya min-CVRR100 control mean", 1.27, cc.mean(), tol=0.02, fmt="{:.2f}")
    chk("Nagoya min-CVRR100 PD mean", 0.71, cp.mean(), tol=0.02, fmt="{:.2f}")
    a, _ = auc(cc, cp)
    chk("Nagoya min-CVRR100 AUC", 0.901, a, tol=0.004)
    a2, _ = auc(nj[nj.Group != "PD"].SDNN200_min, nj[nj.Group == "PD"].SDNN200_min)
    chk("Nagoya min-SDNN200 AUC", 0.919, a2, tol=0.004)

print("\n" + "=" * 96)
print("4. WINDOW-LENGTH LADDER  (Fig. 3D) and CROSS-COHORT (Fig. 3F)")
print("=" * 96)
lad = pd.read_csv(os.path.join(D, "fig7_window_ladder.csv"))
for met, wmin, claim in [("SDNN", 1.4167, 0.905), ("SDNN", 2.8333, 0.919),
                         ("SDNN", 15, 0.795), ("SDNN", 240, 0.760),
                         ("complexity", 15, 0.839), ("complexity", 240, 0.762)]:
    r = lad[(lad.metric == met) & (np.isclose(lad.win_min, wmin, atol=0.01))]
    chk(f"ladder {met} @ {wmin:g} min", claim,
        float(r.AUC.iloc[0]) if len(r) else None, tol=0.004)

h2h = pd.read_csv(os.path.join(D, "fig7_cetram_headtohead.csv"))
al = h2h[h2h.subset == "all"].iloc[0]
ex = h2h[h2h.subset != "all"].iloc[0]
chk("CETRAM min-SDNN, all", 0.480, float(al.SDNN100_min), tol=0.004)
chk("CETRAM complexity, all", 0.655, float(al.complexity), tol=0.004)
chk("CETRAM min-SDNN, excl outliers", 0.466, float(ex.SDNN100_min), tol=0.004)
chk("CETRAM complexity, excl outliers", 0.659, float(ex.complexity), tol=0.004)
chk("CETRAM n, all", 69, int(al.n), tol=0, fmt="{:.0f}")
chk("CETRAM n, excl outliers", 66, int(ex.n), tol=0, fmt="{:.0f}")

print("\n" + "=" * 96)
print("5. INIGUEZ REPLICATION IN CRUCES  (RMSSD, SD1, HF reach nominal p<0.05; none survives /10)")
print("=" * 96)
ini = pd.read_csv(os.path.join(D, "cruces_iniguez_replication.csv"))
for m, pc in [("SDNN", 0.141), ("RMSSD", 0.048), (r"SD1$\times$SD2", 0.062),
              ("HF power", 0.029), ("LF power", 0.068), ("NNiqr", 0.111)]:
    r = ini[ini.metric == m]
    chk(f"Iniguez {m} p", pc, float(r.p.iloc[0]) if len(r) else None, tol=0.0015, fmt="{:.3f}")
r = ini[ini.metric.str.lower() == "nniqr"]
chk("Iniguez NNiqr AUC", 0.632, float(r.AUC.iloc[0]) if len(r) else None, tol=0.004)
mn = ini.p.min()
chk("smallest p across metrics (HF power)", 0.029, mn, tol=0.0015)
chk("no metric survives Bonferroni over 10 (p*10 >= 0.05)", 1, float(mn * 10 >= 0.05), tol=0, fmt="{:.0f}")

print("\n" + "=" * 96)
print("6. RECORDING-LENGTH LADDER  (Discussion: 0.669 at 250 -> 0.852 at 4000 beats)")
print("=" * 96)
ll = pd.read_csv(os.path.join(D, "length_ladder.csv"))
nagll = ll[ll.Cohort.astype(str).str.lower().str.contains("nagoya|japan")]
for N, claim in [(250, 0.669), (4000, 0.852)]:
    s = nagll[nagll.N == N]
    if len(s) == 0:
        chk(f"Nagoya length ladder @ {N} beats", claim, None)
        continue
    a, _ = auc(s[s.Group != "PD"].nAUC, s[s.Group == "PD"].nAUC)
    chk(f"Nagoya length ladder @ {N} beats", claim, a, tol=0.004)

print("\n" + "=" * 96)
print("7. AGE CORRELATIONS  (Results claims seven of eight null)")
print("=" * 96)
age_claims = [("Nagoya 18-22h Control", -0.29, 0.20), ("Nagoya 18-22h PD", -0.48, 0.01)]
nagi = index_by_subject(os.path.join(D, "japan_window_mse.csv"), 1, 20)
agem = ja[["Subject", "Age"]].merge(nagi, on="Subject")
for g, rc, pc in age_claims:
    grp = "Control" if "Control" in g else "PD"
    s = agem[(agem.Group == grp) & agem.Age.notna()]
    if len(s) < 3:
        chk(f"{g} rho", rc, None); continue
    rho, pv = stats.spearmanr(s.Age, s.idx)
    chk(f"{g} rho", rc, rho, tol=0.015, fmt="{:+.2f}")
    chk(f"{g} p", pc, pv, tol=0.008, fmt="{:.2f}")

print("\n" + "=" * 96)
print("8. CROSS-COHORT TRANSFER  (Supplementary Figs. S6-S7, Methods, Discussion)")
print("=" * 96)
xc = pd.read_csv(os.path.join(ROOT, "figures", "FigS6_S7_cross_cohort", "FigS6_S7_stats.csv"))
def xget(design, train, test, model):
    r = xc[(xc.design == design) & (xc.train == train) & (xc.test == test) & (xc.model == model)]
    return float(r.auc.iloc[0]) if len(r) else None
for test, train, cl in [("CETRAM", "Cruces+Nagoya", 0.680), ("Cruces", "CETRAM+Nagoya", 0.688),
                        ("Nagoya", "CETRAM+Cruces", 0.863)]:
    chk(f"LOCO complexity+HR, held-out {test}", cl, xget("LOCO", train, test, "Complexity"))
chk("LOCO SDNN+RMSSD+HR, held-out CETRAM", 0.504, xget("LOCO", "Cruces+Nagoya", "CETRAM", "Conventional"))
for test, train, cl in [("CETRAM", "Cruces+Nagoya", 0.615), ("Cruces", "CETRAM+Nagoya", 0.639),
                        ("Nagoya", "CETRAM+Cruces", 0.745)]:
    chk(f"LOCO HR only, held-out {test}", cl, xget("LOCO", train, test, "HR"))
chk("Transfer Nagoya -> CETRAM", 0.674, xget("single", "Nagoya", "CETRAM", "Complexity"))
chk("Transfer CETRAM -> Nagoya", 0.866, xget("single", "CETRAM", "Nagoya", "Complexity"))

print("\n" + "=" * 96)
print("9. ADDED 2026-11-18 (audit): window scan, primary Nagoya detail, Table 1, flags, sweep, spectra")
print("=" * 96)
ws = pd.read_csv(os.path.join(D, "fig7_window_scan.csv"))
chk("windows with p<0.05 (of 24)", 19, float((ws.p < 0.05).sum()), tol=0, fmt="{:.0f}")
chk("windows surviving Bonferroni", 6, float((ws.p < 0.05 / 24).sum()), tol=0, fmt="{:.0f}")
chk("window-scan AUC min", 0.569, ws.AUC.min(), tol=0.0015)
chk("window-scan AUC max", 0.885, ws.AUC.max(), tol=0.0015)
chk("window-scan AUC median", 0.757, ws.AUC.median(), tol=0.0015)
chk("16-20 h AUC", 0.859, float(ws.set_index("window").AUC[16]), tol=0.0015)
chk("16-20 h rank", 4, float((ws.AUC > ws.set_index("window").AUC[16]).sum() + 1), tol=0, fmt="{:.0f}")
nj = index_by_subject(os.path.join(D, "japan_window_mse.csv"), 1, 20)
c, p_ = nj[nj.Group == "Control"].idx, nj[nj.Group == "PD"].idx
chk("Nagoya control median", 0.733, c.median()); chk("Nagoya PD median", 0.460, p_.median())
jf = pd.read_csv(os.path.join(D, "japan_window_features.csv"))
chk("Table 1 Nagoya PD age mean", 60.3, jf[jf.Group == "PD"].Age.mean(), tol=0.05, fmt="{:.1f}")
chk("Table 1 Nagoya PD age SD", 9.3, jf[jf.Group == "PD"].Age.std(), tol=0.05, fmt="{:.1f}")
a7, p7 = auc(*[g.idx for _, g in index_by_subject(os.path.join(D, "japan_morning_mse.csv"), 1, 20).groupby("Group")])
chk("Nagoya 07-11 h AUC", 0.786, a7)
fl = pd.read_csv(os.path.join(D, "irregular_rhythm_flags.csv"))
for coh, n in (("CETRAM", 3), ("Cruces", 2), ("Nagoya", 6)):
    chk(f"irregular-rhythm flags {coh}", n, float(fl[fl.Cohort == coh].flag.sum()), tol=0, fmt="{:.0f}")
k = nj[~nj.Subject.isin(fl[(fl.Cohort == "Nagoya") & fl.flag].Subject)]
chk("Nagoya AUC without flagged", 0.797, auc(k[k.Group == "Control"].idx, k[k.Group == "PD"].idx)[0])
sp = pd.read_csv(os.path.join(D, "spectral_analysis.csv"))
for coh, b in (("CETRAM", 1.44), ("Cruces", 1.24), ("Nagoya", 0.61)):
    chk(f"spectral beta median {coh}", b, sp[sp.Cohort == coh].beta_beat.median(), tol=0.006, fmt="{:.2f}")
print("\n" + "=" * 96)
print("10. ADDED 2026-11-18 (evening): uncapped 15-min windows, split-half reliability")
print("=" * 96)
ds = pd.read_csv(os.path.join(D, "fig7_dist_summaries.csv")).set_index("stat")
chk("min of 15-min index across day, AUC", 0.839, float(ds.loc["min", "complexity"]))
chk("p10 of 15-min index, AUC", 0.806, float(ds.loc["p10", "complexity"]))
sh = pd.read_csv(os.path.join(D, "split_half_reliability.csv")).dropna()
for c, claim in [("CETRAM (1000 beats)", 0.37), ("Cruces", 0.37), ("Nagoya", 0.42)]:
    g = sh[sh.Cohort == c]
    chk(f"split-half reliability {c}", claim, stats.spearmanr(g.h1, g.h2)[0], tol=0.005, fmt="{:.2f}")

print("\n" + "=" * 96)
print("11. ADDED 2026-11-19: parameter sweep uses the primary template convention")
print("=" * 96)
rs = pd.read_csv(os.path.join(D, "r_sweep.csv"))
rs = rs[(rs.m == 2) & (rs["mode"] == "k") & np.isclose(rs.level, 0.2)]
for coh, f in (("CETRAM", "chile_mse.csv"), ("Cruces", "spain_mse.csv"), ("Nagoya", "japan_window_mse.csv")):
    pm = pd.read_csv(os.path.join(D, f))
    pm = pm[pm.Scales <= 5].sort_values("Scales").groupby("Subject").MSE.apply(
        lambda v: np.trapezoid(v.values, np.arange(1, len(v) + 1)) / len(v))
    g = rs[rs.Cohort == coh].set_index("Subject").cx
    chk(f"sweep k=0.2 m=2 equals primary nAUC(1-5), {coh}: max |diff|", 0.0,
        float((g - pm.reindex(g.index)).abs().max()), tol=1e-9, fmt="{:.1e}")

print("\n" + "=" * 96)
print("CRUCES CURVE SHAPE  (data/cruces_surrogates.csv)")
print("=" * 96)
from scipy.stats import spearmanr as _sp
cs = pd.read_csv(os.path.join(D, "cruces_surrogates.csv"))
_cr, _ce = cs[cs.Cohort == "Cruces"], cs[cs.Cohort == "CETRAM"]
chk("% Cruces curves falling, real", 62, 100 * _cr.falling.mean(), tol=0.6, fmt="{:.0f}")
chk("% Cruces curves falling, IAAFT surrogate", 40, 100 * _cr.falling_surr.mean(), tol=0.6, fmt="{:.0f}")
chk("% CETRAM@500 curves falling, real", 17, 100 * _ce.falling.mean(), tol=0.6, fmt="{:.0f}")
chk("% CETRAM@500 curves falling, IAAFT surrogate", 14, 100 * _ce.falling_surr.mean(), tol=0.6, fmt="{:.0f}")
chk("respiratory-band share, Cruces", 0.41, _cr.resp_share.median(), tol=0.005, fmt="{:.2f}")
chk("respiratory-band share, CETRAM@500", 0.31, _ce.resp_share.median(), tol=0.005, fmt="{:.2f}")
chk("Cruces resp share, falling curves", 0.49, _cr[_cr.falling].resp_share.median(), tol=0.005, fmt="{:.2f}")
chk("Cruces resp share, rising curves", 0.35, _cr[~_cr.falling].resp_share.median(), tol=0.005, fmt="{:.2f}")
chk("Cruces rho(resp peak sharpness, surrogate gap)", 0.14, _sp(_cr.resp_peak, _cr.surr_gap_t5)[0], tol=0.005, fmt="{:+.2f}")

print("\n" + "=" * 96)
print(f"SUMMARY: {CHECKS - len(FAILS)}/{CHECKS} checks passed")
if FAILS:
    print(f"\n{len(FAILS)} MISMATCH(ES):")
    for f in FAILS:
        print(f"   - {f}")
print("=" * 96 + "\n")

if FAILS:
    raise SystemExit(1)
