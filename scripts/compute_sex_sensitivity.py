#!/usr/bin/env python3
"""
Sex as a covariate: cohort composition and sensitivity of the primary comparison.

Answers reviewer comment 3.1 (age and sex adjustment). Reads only data/.

Outputs (data/):
  sex_composition.csv     sex counts by cohort and group, analysed samples (Table 1)
  sex_sensitivity.csv     PD coefficient under nested covariate sets, per cohort
  sex_stratified_auc.csv  AUC of the complexity index within each sex
  sex_missing_scenarios.csv  CETRAM: all assignments of the participants with unknown sex

Index and samples are those of the primary analysis (Fig. 3, Table 3):
  index = trapz(rcMSE curve, tau = 1..n) / n; scales 1-5 (CETRAM, Cruces),
  1-20 (Nagoya, 18:00-22:00 window); HR = 60000 / MeanNN in the same segment.
Models: OLS of the index on group (PD = 1) plus covariates, HC3 standard errors,
complete cases for the covariates in each model (n reported per model).

Sex sources: chile_demographics.csv (M/F), japan_metadata.csv via
japan_window_features.csv (0 = male, 1 = female, as in the Fig. 8 scripts).
Cruces: not available per participant in the harmonized files; the source study
reports 9/21 (43 %) female controls and 11/31 (35 %) female PD (Iniguez et al.),
which is carried into the composition table as source-reported, not analysed.

Run:  python3 scripts/compute_sex_sensitivity.py
"""

import itertools
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")

# Source-reported Cruces sex counts (female, total) — not per participant.
CRUCES_SOURCE_FEMALE = {"Control": (9, 21), "PD": (11, 31)}


# --------------------------------------------------------------------------- data
def index_table(fname, scales):
    m = pd.read_csv(os.path.join(D, fname))
    m["Group"] = m["Group"].str.strip().replace({"control": "Control", "pd": "PD"})
    m = m[m["Group"].isin(["Control", "PD"]) & m["Scales"].isin(list(scales))]
    n = len(list(scales))

    def _idx(g):
        g = g.sort_values("Scales")
        return float(np.trapezoid(g["MSE"].to_numpy(float), g["Scales"].to_numpy(float)) / n)

    return (m.groupby(["Subject", "Group"]).apply(_idx, include_groups=False)
             .rename("Index").reset_index())


def load_cohorts():
    # CETRAM
    c = index_table("chile_mse.csv", range(1, 6))
    met = pd.read_csv(os.path.join(D, "chile_metrics.csv"))
    met["HR"] = 60000.0 / met["HRV_MeanNN"]
    dem = pd.read_csv(os.path.join(D, "chile_demographics.csv")).rename(columns={"Anon_ID": "Subject"})
    c = (c.merge(met[["Subject", "HR"]], on="Subject")
          .merge(dem[["Subject", "Age", "Gender"]], on="Subject", how="left"))
    c["Sex"] = c["Gender"].map({"M": "M", "F": "F"})

    # Cruces (sex not available)
    s = index_table("spain_mse.csv", range(1, 6))
    met = pd.read_csv(os.path.join(D, "spain_metrics.csv"))
    met["HR"] = 60000.0 / met["HRV_MeanNN"]
    dem = pd.read_csv(os.path.join(D, "spain_demographics.csv"))
    s = (s.merge(met[["Subject", "HR"]], on="Subject")
          .merge(dem[["Subject", "Age"]], on="Subject", how="left"))
    s["Sex"] = np.nan

    # Nagoya, 18:00-22:00
    j = index_table("japan_window_mse.csv", range(1, 21))
    f = pd.read_csv(os.path.join(D, "japan_window_features.csv"))
    j = j.merge(f[["Subject", "HR", "Age", "Gender"]], on="Subject")
    j["Sex"] = j["Gender"].map({0: "M", 1: "F"})

    out = {}
    for name, d in (("CETRAM", c), ("Cruces", s), ("Nagoya", j)):
        d = d[["Subject", "Group", "Index", "HR", "Age", "Sex"]].copy()
        d["PD"] = (d["Group"] == "PD").astype(int)
        out[name] = d
    return out


# ----------------------------------------------------------------------- analyses
def composition(cohorts):
    rows = []
    for name, d in cohorts.items():
        for g in ("Control", "PD"):
            x = d[d["Group"] == g]
            if name == "Cruces":
                nf, nt = CRUCES_SOURCE_FEMALE[g]
                rows.append(dict(Cohort=name, Group=g, N=len(x), Female=nf, Male=nt - nf,
                                 Unknown=0, Source="source publication"))
            else:
                rows.append(dict(Cohort=name, Group=g, N=len(x),
                                 Female=int((x["Sex"] == "F").sum()),
                                 Male=int((x["Sex"] == "M").sum()),
                                 Unknown=int(x["Sex"].isna().sum()), Source="per participant"))
    t = pd.DataFrame(rows)
    t["Pct_female_known"] = 100 * t["Female"] / (t["Female"] + t["Male"])
    p = {}
    for name in t["Cohort"].unique():
        x = t[t["Cohort"] == name].set_index("Group")[["Female", "Male"]].to_numpy()
        p[name] = fisher_exact(x)[1]
    t["Fisher_p"] = t["Cohort"].map(p)
    return t


MODELS = [
    ("Crude",                  "Index ~ PD",                 []),
    ("HR (primary)",           "Index ~ PD + HR",            ["HR"]),
    ("HR + age",               "Index ~ PD + HR + Age",      ["HR", "Age"]),
    ("HR + sex",               "Index ~ PD + HR + Sex",      ["HR", "Sex"]),
    ("HR + age + sex",         "Index ~ PD + HR + Age + Sex", ["HR", "Age", "Sex"]),
    ("HR (complete cases)",    "Index ~ PD + HR",            ["HR", "Age", "Sex"]),
]


def sensitivity(cohorts):
    rows = []
    for name, d in cohorts.items():
        for label, formula, need in MODELS:
            x = d.dropna(subset=["Index"] + need)
            if "Sex" in need and x["Sex"].nunique() < 2:
                continue
            m = smf.ols(formula, x).fit(cov_type="HC3")
            lo, hi = m.conf_int().loc["PD"]
            rows.append(dict(Cohort=name, Model=label, n=len(x),
                             n_Control=int((x["PD"] == 0).sum()), n_PD=int((x["PD"] == 1).sum()),
                             beta_PD=m.params["PD"], CI_low=lo, CI_high=hi,
                             beta_PD_SD=m.params["PD"] / x["Index"].std(), p=m.pvalues["PD"]))
        # sex x group interaction (fully adjusted)
        x = d.dropna(subset=["Index", "HR", "Age", "Sex"])
        if x["Sex"].nunique() == 2:
            m = smf.ols("Index ~ PD * Sex + HR + Age", x).fit(cov_type="HC3")
            k = [k for k in m.params.index if ":" in k][0]
            rows.append(dict(Cohort=name, Model="PD x sex interaction", n=len(x),
                             n_Control=int((x["PD"] == 0).sum()), n_PD=int((x["PD"] == 1).sum()),
                             beta_PD=m.params[k], CI_low=m.conf_int().loc[k, 0],
                             CI_high=m.conf_int().loc[k, 1], beta_PD_SD=np.nan, p=m.pvalues[k]))
    return pd.DataFrame(rows)


def stratified_auc(cohorts):
    rows = []
    for name, d in cohorts.items():
        if d["Sex"].isna().all():
            continue
        for sx, lab in (("M", "Male"), ("F", "Female")):
            x = d[d["Sex"] == sx]
            c, p = x[x["PD"] == 0]["Index"], x[x["PD"] == 1]["Index"]
            rows.append(dict(Cohort=name, Sex=lab, n_Control=len(c), n_PD=len(p),
                             AUC=roc_auc_score(x["PD"], -x["Index"]),
                             p_MWU=mannwhitneyu(c, p, alternative="two-sided").pvalue))
        # sex difference within controls, HR-adjusted
        x = d[(d["PD"] == 0)].dropna(subset=["Sex"])
        m = smf.ols("Index ~ Sex + HR", x).fit(cov_type="HC3")
        rows.append(dict(Cohort=name, Sex="Controls: male - female (HR-adj.)",
                         n_Control=len(x), n_PD=0, AUC=m.params["Sex[T.M]"],
                         p_MWU=m.pvalues["Sex[T.M]"]))
        # how well sex alone separates the groups
        x = d.dropna(subset=["Sex"])
        rows.append(dict(Cohort=name, Sex="Sex alone as classifier (male = PD)",
                         n_Control=int((x["PD"] == 0).sum()), n_PD=int((x["PD"] == 1).sum()),
                         AUC=roc_auc_score(x["PD"], (x["Sex"] == "M").astype(int)), p_MWU=np.nan))
    return pd.DataFrame(rows)


def missing_scenarios(d):
    miss = d.loc[d["Sex"].isna(), "Subject"].tolist()
    rows = []
    for combo in itertools.product("MF", repeat=len(miss)):
        x = d.copy()
        for sid, sx in zip(miss, combo):
            x.loc[x["Subject"] == sid, "Sex"] = sx
        m1 = smf.ols("Index ~ PD + HR + Sex", x).fit(cov_type="HC3")
        xa = x.dropna(subset=["Age"])
        m2 = smf.ols("Index ~ PD + HR + Age + Sex", xa).fit(cov_type="HC3")
        tab = pd.crosstab(x["Group"], x["Sex"])[["F", "M"]].to_numpy()
        rows.append(dict(Assignment=",".join(f"{s}={v}" for s, v in zip(miss, combo)),
                         Fisher_p=fisher_exact(tab)[1],
                         beta_HR_sex=m1.params["PD"], p_HR_sex=m1.pvalues["PD"],
                         n_full=len(xa), beta_HR_age_sex=m2.params["PD"],
                         p_HR_age_sex=m2.pvalues["PD"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main
def main(write=True):
    cohorts = load_cohorts()
    comp = composition(cohorts)
    sens = sensitivity(cohorts)
    strat = stratified_auc(cohorts)
    scen = missing_scenarios(cohorts["CETRAM"])

    pd.set_option("display.width", 160)
    print("\nSex composition of the analysed samples\n", comp.round(3).to_string(index=False))
    print("\nPD coefficient under nested covariate sets (OLS, HC3)\n", sens.round(4).to_string(index=False))
    print("\nWithin-sex discrimination and sex effects\n", strat.round(3).to_string(index=False))
    print("\nCETRAM: every assignment of unknown sex\n", scen.round(3).to_string(index=False))

    if write:
        comp.to_csv(os.path.join(D, "sex_composition.csv"), index=False)
        sens.to_csv(os.path.join(D, "sex_sensitivity.csv"), index=False)
        strat.to_csv(os.path.join(D, "sex_stratified_auc.csv"), index=False)
        scen.to_csv(os.path.join(D, "sex_missing_scenarios.csv"), index=False)
    return comp, sens, strat, scen


if __name__ == "__main__":
    import sys
    main(write="--no-write" not in sys.argv)
