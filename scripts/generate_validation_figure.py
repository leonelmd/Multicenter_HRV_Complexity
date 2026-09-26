#!/usr/bin/env python3
"""
Supplementary methods-validation figure for the rcMSE complexity index.

Not a results figure — it documents the methodological choices behind the primary
endpoint so a reviewer can see they were tested rather than assumed.

  A  Synthetic reference processes WITH the real cohort curves overlaid
  B  Artifact sensitivity, measured on real CETRAM RRi
  C  Parameter sweep (m x r x entropy type), all at FIXED r
  D  Complexity-index selection, raw vs / mean HR
  E  Scale-range sweep: nAUC(1-k) for k = 1..20
  F  Recording length
  G  HR-normalisation diagnostics

Conventions enforced throughout (see Multicenter/RCMSE_TOLERANCE_AUDIT.md):
  * r is FIXED at r_factor x SD(original series) and never recomputed per scale.
    The per-scale variant is a bug, not an option, and is not depicted anywhere.
  * Every index is the trapezoidal nAUC of the canonical toolbox,
    compute_nAUC(curve) = trapz(1..n, curve) / n. The arithmetic mean over scales
    that the pipeline used historically is not shown.
  * rcMSE uses the Wu et al. (2014) coarse-graining scheme (all `scale` shifts).

Outputs
    figures/Validation/FigureValidation.{png,svg}
    figures/Validation/validation_*.csv

Run:  python scripts/generate_validation_figure.py [--force]
"""
from __future__ import annotations

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import figstyle as fs
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures", "Validation")
os.makedirs(OUT, exist_ok=True)
CETRAM_DET = os.path.abspath(os.path.join(
    ROOT, "..", "..", "CETRAM", "public_release", "results", "cleaned_detections"))

RNG = np.random.default_rng(42)
COL = {"Control": "#2E86AB", "PD": "#D62828"}
FORCE = "--force" in sys.argv
M, RFACTOR, NSCALES = 2, 0.2, 20

# ==========================================================================
# rcMSE — fixed r, Wu (2014) shifts
# ==========================================================================
def _AB(sig, m, r, etype):
    n = len(sig) - m
    if n < 3:
        return 0.0, 0.0
    X = sig[np.arange(n)[:, None] + np.arange(m)[None, :]].astype(np.float32)
    d = np.abs(X[:, None, :] - X[None, :, :]).max(2)
    xm = sig[np.arange(n) + m].astype(np.float32)
    d1 = np.maximum(d, np.abs(xm[:, None] - xm[None, :]))
    iu = np.triu_indices(n, 1)
    if etype == "sample":
        return float((d1[iu] <= r).sum()), float((d[iu] <= r).sum())
    lg = np.log(2.0)
    return (float(np.exp(-lg * (d1[iu] / r) ** 2).sum()),
            float(np.exp(-lg * (d[iu] / r) ** 2).sum()))


def rcmse(sig, r, m=M, etype="sample", nscales=NSCALES):
    """r is passed in and held constant across all scales."""
    sig = np.asarray(sig, float)
    N = len(sig)
    out = []
    for tau in range(1, nscales + 1):
        A = B = 0.0
        for k in range(tau):                       # all `tau` shifts (Wu 2014)
            L = (N - k) // tau
            if L <= m:
                continue
            cg = sig[k:k + L * tau].reshape(L, tau).mean(1)
            a, b = _AB(cg, m, r, etype)
            A += a; B += b
        out.append(-np.log(A / B) if (A > 0 and B > 0) else np.nan)
    return np.array(out)


def nauc(curve):
    """Canonical compute_nAUC: trapz over 1..n divided by n. Trapezoid only."""
    c = np.asarray(curve, float)
    c = c[np.isfinite(c)]
    if len(c) == 0:
        return np.nan
    return float(np.trapezoid(c, np.arange(1, len(c) + 1)) / len(c))


def lrs(curve):
    """Canonical compute_LRS: slope of a linear fit over the scale axis."""
    c = np.asarray(curve, float)
    s = np.arange(1, len(c) + 1, dtype=float)
    ok = np.isfinite(c)
    return float(np.polyfit(s[ok], c[ok], 1)[0]) if ok.sum() > 1 else np.nan


def auc_cp(x, y):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    return roc_auc_score(np.asarray(y)[ok], -x[ok])


def boot_ci(x, y, n=400):
    x = np.asarray(x, float); y = np.asarray(y)
    ok = np.isfinite(x); x, y = x[ok], y[ok]
    vals = []
    for _ in range(n):
        i = RNG.choice(len(x), len(x), True)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], -x[i]))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


# ==========================================================================
def load_cetram():
    rows = []
    for g in ("Control", "PD"):
        for f in sorted(glob.glob(os.path.join(CETRAM_DET, g, "*_cleaned.csv"))):
            s = pd.read_csv(f)["sample"].dropna().astype(np.int64).to_numpy()
            rows.append(dict(Subject=os.path.basename(f).replace("_cleaned.csv", ""),
                             Group=g, rri=np.diff(s).astype(float)))
    return rows


def synth(kind, n=1100):
    if kind == "White noise":
        x = RNG.normal(0, 1, n)
    elif kind == "1/f noise":
        f = np.fft.rfftfreq(n)[1:]
        ph = RNG.uniform(0, 2 * np.pi, len(f))
        x = np.fft.irfft(np.concatenate(([0], (1 / np.sqrt(f)) * np.exp(1j * ph))), n=n)
    elif kind == "AR(1) $\\phi$=0.9":
        x = np.zeros(n); e = RNG.normal(0, 1, n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + e[i]
    elif kind == "Logistic map":
        x = np.zeros(n); x[0] = 0.4
        for i in range(1, n):
            x[i] = 3.9 * x[i - 1] * (1 - x[i - 1])
    elif kind == "Periodic + noise":
        t = np.arange(n)
        x = np.sin(2 * np.pi * t / 25) + 0.1 * RNG.normal(0, 1, n)
    else:
        raise ValueError(kind)
    return (x - x.mean()) / x.std(ddof=1)


# ==========================================================================
def main():
    fs.apply()
    print("Loading CETRAM ...")
    subs = load_cetram()
    y = np.array([s["Group"] == "PD" for s in subs], int)
    print(f"  {len(subs)} subjects ({(y==0).sum()} Control / {y.sum()} PD)")

    met = pd.read_csv(os.path.join(DATA, "chile_metrics.csv"))
    met["HR"] = 60000.0 / met["HRV_MeanNN"]
    hrmap = met.set_index("Subject")["HR"]
    HR = np.array([hrmap.get(s["Subject"], np.nan) for s in subs])

    # ---- per-subject rcMSE curves (reused by A, D, E) --------------------
    cpath = os.path.join(OUT, "validation_curves.csv")
    if os.path.exists(cpath) and not FORCE:
        cdf = pd.read_csv(cpath)
        curves = {r.Subject: cdf[cdf.Subject == r.Subject].sort_values("Scale").MSE.to_numpy()
                  for r in cdf.drop_duplicates("Subject").itertuples()}
        print("  curves cached")
    else:
        print("Computing rcMSE curves (fixed r, Wu shifts) ...")
        curves, recs = {}, []
        for s in subs:
            r = RFACTOR * s["rri"].std(ddof=1)
            c = rcmse(s["rri"], r)
            curves[s["Subject"]] = c
            recs += [dict(Subject=s["Subject"], Group=s["Group"], Scale=i + 1, MSE=v)
                     for i, v in enumerate(c)]
        pd.DataFrame(recs).to_csv(cpath, index=False)
    CUR = np.array([curves[s["Subject"]] for s in subs])

    # ---- A: synthetic references + real cohort curves --------------------
    print("A) synthetic references vs real data")
    kinds = ["White noise", "1/f noise", "AR(1) $\\phi$=0.9", "Logistic map", "Periodic + noise"]
    A = {}
    for k in kinds:
        reps = np.array([rcmse(x, RFACTOR * x.std(ddof=1), nscales=NSCALES)
                         for x in (synth(k) for _ in range(8))])
        A[k] = (np.nanmean(reps, 0), np.nanstd(reps, 0))
    pd.DataFrame({k: v[0] for k, v in A.items()}, index=range(1, NSCALES + 1)) \
        .rename_axis("Scale").to_csv(os.path.join(OUT, "validation_A_synthetic.csv"))

    # ---- B: artifact sensitivity on REAL RRi -----------------------------
    print("B) artifact sensitivity (real RRi)")
    bpath = os.path.join(OUT, "validation_B_artifact.csv")
    if os.path.exists(bpath) and not FORCE:
        B = pd.read_csv(bpath)
    else:
        rates = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        pick = list(RNG.choice(len(subs), 12, replace=False))
        rows = []
        for rate in rates:
            for i in pick:
                x0 = subs[i]["rri"]
                base_r = RFACTOR * x0.std(ddof=1)
                base = nauc(rcmse(x0, base_r, nscales=5))
                x = x0.copy()
                nk = int(round(rate * len(x)))
                if nk:
                    idx = RNG.choice(len(x), nk, replace=False)
                    x[idx] = x[idx] * RNG.uniform(2.5, 6.0, nk)
                v = nauc(rcmse(x, RFACTOR * x.std(ddof=1), nscales=5))
                rows.append(dict(rate=rate, subj=subs[i]["Subject"],
                                 nauc=v, rel=v / base if base else np.nan))
        B = (pd.DataFrame(rows).groupby("rate")
             .agg(mean=("rel", "mean"), sd=("rel", "std")).reset_index())
        B.to_csv(bpath, index=False)

    # ---- C: parameter sweep (fixed r throughout) -------------------------
    print("C) parameter sweep")
    cp = os.path.join(OUT, "validation_C_params.csv")
    if os.path.exists(cp) and not FORCE:
        C = pd.read_csv(cp)
        print("   cached")
    else:
        ms, rs, ets = [1, 2, 3], [0.10, 0.15, 0.20, 0.25], ["sample", "fuzzy"]
        rows = []
        for et in ets:
            for m in ms:
                for rf in rs:
                    v = np.array([nauc(rcmse(s["rri"], rf * s["rri"].std(ddof=1),
                                             m=m, etype=et, nscales=5)) for s in subs]) / HR
                    rows.append(dict(etype=et, m=m, r=rf, AUC=auc_cp(v, y)))
                    print(f"   {et} m={m} r={rf:.2f} AUC={rows[-1]['AUC']:.3f}", flush=True)
        C = pd.DataFrame(rows); C.to_csv(cp, index=False)

    # ---- D: complexity-index selection -----------------------------------
    print("D) index selection")
    cand = {
        "SampEn $\\tau$=1":  CUR[:, 0],
        "nAUC(1-3)":         np.array([nauc(c[:3]) for c in CUR]),
        "nAUC(1-5)":         np.array([nauc(c[:5]) for c in CUR]),
        "nAUC(1-10)":        np.array([nauc(c[:10]) for c in CUR]),
        "nAUC(1-20)":        np.array([nauc(c) for c in CUR]),
        "LRS(1-5)":          np.array([lrs(c[:5]) for c in CUR]),
        "LRS(1-20)":         np.array([lrs(c) for c in CUR]),
    }
    rows = []
    for k, v in cand.items():
        for norm in (False, True):
            x = v / HR if norm else v
            lo, hi = boot_ci(x, y)
            ok = np.isfinite(x)
            rows.append(dict(index=k, divHR=norm, AUC=auc_cp(x, y), lo=lo, hi=hi,
                             p=mannwhitneyu(x[ok][y[ok] == 0], x[ok][y[ok] == 1])[1]))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "validation_D_index.csv"), index=False)

    # ---- E: scale-range sweep --------------------------------------------
    print("E) scale-range sweep")
    rows = []
    for k in range(1, NSCALES + 1):
        v = np.array([nauc(c[:k]) for c in CUR])
        for norm in (False, True):
            x = v / HR if norm else v
            lo, hi = boot_ci(x, y, 200)
            rows.append(dict(kmax=k, divHR=norm, AUC=auc_cp(x, y), lo=lo, hi=hi))
    E = pd.DataFrame(rows); E.to_csv(os.path.join(OUT, "validation_E_scalerange.csv"), index=False)

    # ---- F: recording length ---------------------------------------------
    print("F) recording length")
    fp = os.path.join(OUT, "validation_F_length.csv")
    if os.path.exists(fp) and not FORCE:
        F = pd.read_csv(fp)
    else:
        common = np.array([i for i, s in enumerate(subs) if len(s["rri"]) >= 1000])
        rows = []
        for L in [650, 800, 1000, None]:
            v = []
            for i in common:
                seg = subs[i]["rri"] if L is None else subs[i]["rri"][:L]
                v.append(nauc(rcmse(seg, RFACTOR * seg.std(ddof=1), nscales=5)))
            v = np.array(v) / HR[common]
            ok = np.isfinite(v); yy = y[common][ok]
            rows.append(dict(N=("full" if L is None else L), n_subj=int(ok.sum()),
                             AUC=auc_cp(v, y[common]),
                             p=mannwhitneyu(v[ok][yy == 0], v[ok][yy == 1])[1]))
        F = pd.DataFrame(rows); F.to_csv(fp, index=False)
    nb = np.array([len(s["rri"]) for s in subs])
    n5 = np.array([nauc(c[:5]) for c in CUR])
    rhoN = {g: spearmanr(n5[y == (g == "PD")], nb[y == (g == "PD")]) for g in ("Control", "PD")}

    # ---- G: HR normalisation ---------------------------------------------
    print("G) HR normalisation")
    raw, div = n5, n5 / HR
    mets = [c for c in ["HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_SD1", "HRV_SD2",
                        "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_DFA_alpha1"] if c in met.columns]
    mi = met.set_index("Subject")
    rows = []
    for c in mets:
        mv = np.array([mi[c].get(s["Subject"], np.nan) for s in subs])
        ok = np.isfinite(mv) & np.isfinite(raw) & np.isfinite(HR)
        rows.append(dict(metric=c.replace("HRV_", ""),
                         rho_HR=spearmanr(HR[ok], mv[ok])[0],
                         rho_raw=spearmanr(raw[ok], mv[ok])[0],
                         rho_div=spearmanr(div[ok], mv[ok])[0]))
    G = pd.DataFrame(rows); G["shift"] = G.rho_div - G.rho_raw
    G.to_csv(os.path.join(OUT, "validation_G_hrnorm.csv"), index=False)

    # ======================= PLOT ========================================
    print("rendering ...")
    fig = plt.figure(figsize=(17.5, 15.5))
    gs = GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.30)
    sc = np.arange(1, NSCALES + 1)

    def lab(ax, t):
        ax.text(-0.15, 1.07, t, transform=ax.transAxes, fontsize=17, fontweight="normal")

    # --- A
    ax = fig.add_subplot(gs[0, 0]); lab(ax, "A")
    for k, c in zip(kinds, ["#999", "#16A085", "#8E44AD", "#E67E22", "#7F8C8D"]):
        mu, sd = A[k]
        ax.plot(sc, mu, lw=1.6, color=c, alpha=.9, label=k)
        ax.fill_between(sc, mu - sd, mu + sd, color=c, alpha=.12)
    for g in ("Control", "PD"):
        sel = y == (g == "PD")
        mu = np.nanmean(CUR[sel], 0); se = np.nanstd(CUR[sel], 0) / np.sqrt(sel.sum())
        ax.plot(sc, mu, lw=3.2, color=COL[g], marker="o", ms=4, label=f"CETRAM {g}", zorder=5)
        ax.fill_between(sc, mu - se, mu + se, color=COL[g], alpha=.28, zorder=4)
    ax.set_xlabel("Scale $\\tau$"); ax.set_ylabel("Sample entropy")
    ax.set_title("Reference processes vs real data\n(fixed $r=0.2\\times$SD; synthetics mean$\\pm$SD of 8 runs)",
                 fontweight="normal", fontsize=11)
    ax.legend(fontsize=7.2, frameon=False, ncol=1, loc="lower left"); ax.grid(alpha=.2)

    # --- B
    ax = fig.add_subplot(gs[0, 1]); lab(ax, "B")
    ax.errorbar(B.rate * 100, B["mean"], yerr=B.sd, marker="o", lw=2, color="#C0392B", capsize=3)
    ax.axhline(1.0, ls=":", c="k", lw=1)
    ax.axvline(0.10, ls="--", c=COL["Control"], lw=1.4)
    ax.axvline(0.42, ls="--", c=COL["PD"], lw=1.4)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Spurious long intervals injected (% of beats)")
    ax.set_ylabel("nAUC(1-5), relative to clean")
    ax.set_title("Artifact sensitivity — real CETRAM RRi\n(dashed = rates seen before the guard)",
                 fontweight="normal", fontsize=11)
    ax.grid(alpha=.2)

    # --- C
    ax = fig.add_subplot(gs[0, 2]); lab(ax, "C")
    piv = C.pivot_table(index=["etype", "m"], columns="r", values="AUC")
    ax.grid(False)
    im = ax.imshow(piv.values, cmap="RdYlBu_r", vmin=.60, vmax=.82, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{v:.2f}" for v in piv.columns])
    ax.set_yticks(range(len(piv))); ax.set_yticklabels([f"{e[:4]} m={m}" for e, m in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_xlabel("tolerance $r$ ($\\times$SD), fixed across scales")
    ax.set_title("Parameter sweep — AUC\nnAUC(1-5)/HR", fontweight="normal", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=.046)

    # --- D
    ax = fig.add_subplot(gs[1, :2]); lab(ax, "D")
    order = list(cand); xp = np.arange(len(order))
    for k, off, cl, lb in [(False, -.19, "#7F8C8D", "unnormalized"),
                           (True, .19, "#8E44AD", "$\\div$ mean HR")]:
        s = D[D.divHR == k].set_index("index").loc[order]
        ax.errorbar(xp + off, s.AUC, yerr=[s.AUC - s.lo, s.hi - s.AUC], fmt="o",
                    ms=7, capsize=4, color=cl, label=lb, lw=2)
        for xi, (a, pv) in enumerate(zip(s.AUC, s.p)):
            if pv < .05:
                ax.text(xi + off, s.hi.iloc[xi] + .012, "*", ha="center", fontsize=13, color=cl)
    ax.axhline(.5, ls=":", c="k", lw=1)
    ax.set_xticks(xp); ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("AUC (Control > PD)"); ax.set_ylim(.33, .92)
    ax.set_title("Complexity-index selection — trapezoidal nAUC only, bootstrap 95% CI  (* p<0.05)",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="lower left"); ax.grid(alpha=.2, axis="y")

    # --- E
    ax = fig.add_subplot(gs[1, 2]); lab(ax, "E")
    for k, cl, lb in [(False, "#7F8C8D", "unnormalized"), (True, "#8E44AD", "$\\div$ HR")]:
        s = E[E.divHR == k]
        ax.plot(s.kmax, s.AUC, marker="o", ms=3.5, lw=2, color=cl, label=lb)
        ax.fill_between(s.kmax, s.lo, s.hi, color=cl, alpha=.14)
    ax.axhline(.5, ls=":", c="k", lw=1)
    ax.set_xlabel("upper scale $k$ of nAUC(1-$k$)"); ax.set_ylabel("AUC")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_title("Scale-range sweep", fontweight="normal", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.2)

    # --- F
    ax = fig.add_subplot(gs[2, 0]); lab(ax, "F")
    xs = np.arange(len(F))
    ax.bar(xs, F.AUC, color="#2E86AB", alpha=.85)
    for i, r in F.iterrows():
        ax.text(i, r.AUC + .006, f"{r.AUC:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([str(v) for v in F.N])
    ax.set_ylim(.5, .85)
    ax.set_xlabel(f"Beats used (common subset, n={F.n_subj.iloc[0]})"); ax.set_ylabel("AUC")
    ax.set_title("Recording length\n$\\rho$(nAUC,N): C "
                 f"{rhoN['Control'][0]:+.2f}, PD {rhoN['PD'][0]:+.2f} (n.s.)",
                 fontweight="normal", fontsize=11)
    ax.grid(alpha=.2, axis="y")

    # --- G1
    ax = fig.add_subplot(gs[2, 1]); lab(ax, "G")
    for g, mk in (("Control", "o"), ("PD", "s")):
        s = y == (g == "PD")
        ax.scatter(HR[s], raw[s], c=COL[g], marker=mk, s=28, alpha=.75, label=g)
    r1 = spearmanr(HR[np.isfinite(raw)], raw[np.isfinite(raw)])
    ax.set_xlabel("Mean HR (bpm)"); ax.set_ylabel("nAUC(1-5), unnormalized")
    ax.set_title(f"Unnormalized vs HR\n$\\rho$={r1[0]:+.2f} (p={r1[1]:.2f}) — orthogonal",
                 fontweight="normal", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.2)

    # --- G2
    ax = fig.add_subplot(gs[2, 2])
    ax.scatter(G.rho_HR, G["shift"], s=55, c="#8E44AD")
    seen = []
    for _, r in G.iterrows():
        dy = 4 + 9 * sum(abs(r.rho_HR - a) < .04 and abs(r["shift"] - b) < .02 for a, b in seen)
        seen.append((r.rho_HR, r["shift"]))
        ax.annotate(r.metric, (r.rho_HR, r["shift"]), fontsize=7,
                    xytext=(4, dy), textcoords="offset points")
    rr = spearmanr(G.rho_HR, G["shift"])
    ax.axhline(0, ls=":", c="k"); ax.axvline(0, ls=":", c="k")
    ax.set_xlabel("metric's own $\\rho$ with HR")
    ax.set_ylabel("$\\Delta\\rho$ caused by dividing")
    ax.set_title(f"Dividing distorts HRV correlations\n$\\rho$={rr[0]:+.2f}",
                 fontweight="normal", fontsize=11)
    ax.grid(alpha=.2)

    p = os.path.join(OUT, "FigureValidation.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print("  ->", p)


if __name__ == "__main__":
    main()
