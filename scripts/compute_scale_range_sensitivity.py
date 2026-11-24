#!/usr/bin/env python3
"""
Scale-range sensitivity of the complexity index, and the number of participants that
meet the N/tau >= 200 guideline at each scale (Supplementary Table "scale range").

Reads only data/. Index as in the primary analysis (trapz(rcMSE, tau = a..b) / n),
here for sample entropy at tau = 1, nAUC 1-2 and nAUC 1-5 in every cohort, and nAUC
1-20 in Nagoya (18:00-22:00 window). HR = 60000 / MeanNN in the same segment.
Models: OLS of the index on group plus covariates, HC3 standard errors, complete cases.

Recording lengths N (retained intervals): CETRAM from length_ladder.csv (N_full),
Cruces and Nagoya from split_half_reliability.csv (N, the analysed series).

Also reports the Nagoya 15-min windows with 600-999 beats, and the min/median
across-day AUC when those windows are dropped.

Outputs (data/):
  scale_range_sensitivity.csv   AUC, p and adjusted p per cohort and index
  scale_range_counts.csv        participants meeting N/tau >= 200, per scale and group

Run:  python3 scripts/compute_scale_range_sensitivity.py
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")


def index_table(fname, scales):
    m = pd.read_csv(os.path.join(D, fname))
    m["Group"] = m["Group"].str.strip().replace({"control": "Control", "pd": "PD"})
    m = m[m["Group"].isin(["Control", "PD"]) & m["Scales"].isin(list(scales))]
    n = len(list(scales))

    def _idx(g):
        g = g.sort_values("Scales")
        if n == 1:
            return float(g["MSE"].iloc[0])
        return float(np.trapezoid(g["MSE"].to_numpy(float), g["Scales"].to_numpy(float)) / n)

    return (m.groupby(["Subject", "Group"]).apply(_idx, include_groups=False)
             .rename("Index").reset_index())


def covariates(cohort):
    if cohort == "CETRAM":
        met = pd.read_csv(os.path.join(D, "chile_metrics.csv"))
        met["HR"] = 60000.0 / met["HRV_MeanNN"]
        dem = pd.read_csv(os.path.join(D, "chile_demographics.csv")).rename(columns={"Anon_ID": "Subject"})
        c = met[["Subject", "HR"]].merge(dem[["Subject", "Age", "Gender"]], on="Subject", how="left")
        c["Sex"] = c["Gender"].map({"M": "M", "F": "F"})
        return c[["Subject", "HR", "Age", "Sex"]], "chile_mse.csv"
    if cohort == "Cruces":
        met = pd.read_csv(os.path.join(D, "spain_metrics.csv"))
        met["HR"] = 60000.0 / met["HRV_MeanNN"]
        dem = pd.read_csv(os.path.join(D, "spain_demographics.csv"))
        c = met[["Subject", "HR"]].merge(dem[["Subject", "Age"]], on="Subject", how="left")
        c["Sex"] = np.nan            # not available per participant
        return c, "spain_mse.csv"
    f = pd.read_csv(os.path.join(D, "japan_window_features.csv"))
    f["Sex"] = f["Gender"].map({0: "M", 1: "F"})
    return f[["Subject", "HR", "Age", "Sex"]], "japan_window_mse.csv"


def contrast(d):
    d = d.copy()
    d["PD"] = (d["Group"] == "PD").astype(int)
    auc = roc_auc_score(d["PD"], -d["Index"])                 # > 0.5: lower in PD
    p = mannwhitneyu(d.loc[d.PD == 0, "Index"], d.loc[d.PD == 1, "Index"]).pvalue
    m = smf.ols("Index ~ PD + HR", d).fit(cov_type="HC3")
    auc_adj = roc_auc_score(d["PD"], -(d["Index"] - m.params["HR"] * d["HR"]))
    da = d.dropna(subset=["Age"])
    p_age = smf.ols("Index ~ PD + HR + Age", da).fit(cov_type="HC3").pvalues["PD"]
    ds = d.dropna(subset=["Age", "Sex"])
    p_sex = (smf.ols("Index ~ PD + HR + Age + Sex", ds).fit(cov_type="HC3").pvalues["PD"]
             if len(ds) > 10 and ds["Sex"].nunique() > 1 else np.nan)
    return dict(n=len(d), n_Control=int((d.PD == 0).sum()), n_PD=int((d.PD == 1).sum()),
                AUC=auc, p=p, AUC_HR_adj=auc_adj, p_HR=m.pvalues["PD"],
                p_HR_age=p_age, p_HR_age_sex=p_sex)


def lengths():
    L = pd.read_csv(os.path.join(D, "length_ladder.csv"))
    ce = (L[L.Cohort == "CETRAM"].groupby("Subject")["N_full"].first()
          .rename("N").reset_index())
    S = pd.read_csv(os.path.join(D, "split_half_reliability.csv"))
    out = {}
    for cohort, src, f in (("CETRAM", ce, "chile_mse.csv"),
                           ("Cruces", S[S.Cohort == "Cruces"][["Subject", "N"]], "spain_mse.csv"),
                           ("Nagoya", S[S.Cohort == "Nagoya"][["Subject", "N"]], "japan_window_mse.csv")):
        g = index_table(f, range(1, 2))[["Subject", "Group"]]
        out[cohort] = g.merge(src, on="Subject")
        assert len(out[cohort]) == len(g), f"{cohort}: missing N for some participants"
    return out


def main():
    rows = []
    for cohort in ("CETRAM", "Cruces", "Nagoya"):
        cov, f = covariates(cohort)
        ranges = [("SampEn tau=1", range(1, 2)), ("nAUC 1-2", range(1, 3)), ("nAUC 1-5", range(1, 6))]
        if cohort == "Nagoya":
            ranges.append(("nAUC 1-20", range(1, 21)))
        for lab, r in ranges:
            d = index_table(f, r).merge(cov, on="Subject")
            rows.append(dict(Cohort=cohort, Index=lab, **contrast(d)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(D, "scale_range_sensitivity.csv"), index=False)

    crow = []
    for cohort, d in lengths().items():
        taus = [1, 2, 3, 4, 5] + ([10, 20] if cohort == "Nagoya" else [])
        for t in taus:
            ok = d["N"] / t >= 200
            crow.append(dict(Cohort=cohort, tau=t, n_total=len(d),
                             meet_Control=int(ok[d.Group == "Control"].sum()),
                             meet_PD=int(ok[d.Group == "PD"].sum()),
                             meet_total=int(ok.sum()),
                             N_median=float(d["N"].median()),
                             N_q1=float(d["N"].quantile(.25)), N_q3=float(d["N"].quantile(.75)),
                             N_min=int(d["N"].min()), N_max=int(d["N"].max())))
    C = pd.DataFrame(crow)
    C.to_csv(os.path.join(D, "scale_range_counts.csv"), index=False)

    # Nagoya 15-min windows below 1000 beats
    keep = set(pd.read_csv(os.path.join(D, "japan_window_features.csv")).Subject)
    w = pd.read_csv(os.path.join(D, "japan_15min_windows.csv"))
    w = w[w.Subject.isin(keep)]
    low = (w.n_beats < 1000)
    print(f"Nagoya 15-min windows: {len(w)}; 600-999 beats: {int(low.sum())} "
          f"({100 * low.mean():.1f}%), in {w[low].Subject.nunique()} participants")
    for lab, ww in (("all windows", w), (">=1000 beats", w[~low])):
        for how in ("min", "median"):
            s = ww.groupby(["Subject", "Group"]).cx.agg(how).reset_index()
            g = (s.Group == "PD").astype(int)
            print(f"  {lab:13s} {how:6s} AUC {roc_auc_score(g, -s.cx):.3f} "
                  f"p {mannwhitneyu(s.cx[g == 0], s.cx[g == 1]).pvalue:.4f} n {len(s)}")

    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))
    print(C[["Cohort", "tau", "meet_Control", "meet_PD", "meet_total", "n_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
