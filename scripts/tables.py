#!/usr/bin/env python3
"""Main-text tables, from data/: figures/Tables/*.csv

  Table1_demographics.csv    n and age (mean, SD) by cohort and group (Table 1)
  Table2_conventional.csv    directional AUC and Mann-Whitney p, complexity and HRV (Table 2)
  Table3_primary.csv         median [IQR], p, AUC with stratified-bootstrap 95% CI (Table 3)
  Table4_hr_adjustment.csv   HR-adjusted AUC, p and standardized group coefficient (Table 4)

AUC = U / (n_C n_PD), oriented so that AUC > 0.5 means lower values in PD (HR: higher in PD).
CI: percentile bootstrap, 4000 resamples within each group, seed 0.
HR adjustment: OLS of the index on group and HR with HC3 errors; adjusted AUC is the AUC of
the index minus its fitted HR term; beta_PD from the same model on the standardized index.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu

from paths import DATA, figdir

COHORTS = ("CETRAM", "Cruces", "Nagoya")


def nauc(v):
    v = np.asarray(v, float)
    return np.trapezoid(v, np.arange(1, len(v) + 1)) / len(v)


def index(fname, top):
    d = pd.read_csv(os.path.join(DATA, fname))
    d = d[d.Group.isin(["Control", "PD"]) & (d.Scales <= top)].sort_values("Scales")
    return d.groupby(["Subject", "Group"]).MSE.apply(lambda v: nauc(v.values)).rename("cx").reset_index()


def auc_p(c, p):
    c, p = np.asarray(c, float), np.asarray(p, float)
    c, p = c[~np.isnan(c)], p[~np.isnan(p)]
    u = mannwhitneyu(c, p)
    return u.statistic / len(c) / len(p), u.pvalue


def boot_ci(c, p, rng, n=4000):
    c, p = np.asarray(c, float), np.asarray(p, float)
    a = [auc_p(rng.choice(c, len(c)), rng.choice(p, len(p)))[0] for _ in range(n)]
    return np.percentile(a, [2.5, 97.5])


def load():
    idx = {"CETRAM": index("chile_mse.csv", 5), "Cruces": index("spain_mse.csv", 5),
           "Nagoya": index("japan_window_mse.csv", 20)}
    ch = pd.read_csv(os.path.join(DATA, "chile_metrics.csv"))
    sp = pd.read_csv(os.path.join(DATA, "spain_metrics.csv"))
    jw = pd.read_csv(os.path.join(DATA, "japan_window_features.csv"))
    for m in (ch, sp):
        m["HR"] = 60000.0 / m.HRV_MeanNN
    jw = jw.rename(columns={"DFA_alpha1": "HRV_DFA_alpha1"})
    met = {"CETRAM": ch, "Cruces": sp, "Nagoya": jw}
    ages = {"CETRAM": pd.read_csv(os.path.join(DATA, "chile_demographics.csv")).rename(columns={"Anon_ID": "Subject"}),
            "Cruces": pd.read_csv(os.path.join(DATA, "spain_demographics.csv")),
            "Nagoya": pd.read_csv(os.path.join(DATA, "japan_metadata.csv")).rename(columns={"Subject_ID": "Subject"})}
    for c in COHORTS:
        idx[c] = (idx[c].merge(met[c].drop(columns=["Group", "Age"], errors="ignore"), on="Subject", how="left")
                  .merge(ages[c][["Subject", "Age"]], on="Subject", how="left"))
    return idx


def main():
    out = figdir("Tables")
    d = load()
    rng = np.random.default_rng(0)
    t1, t2, t3, t4 = [], [], [], []
    for c in COHORTS:
        x = d[c]
        C, P = x[x.Group == "Control"], x[x.Group == "PD"]
        t1.append(dict(Cohort=c, n_Control=len(C), n_PD=len(P),
                       age_Control_mean=C.Age.mean(), age_Control_sd=C.Age.std(),
                       age_PD_mean=P.Age.mean(), age_PD_sd=P.Age.std(), n_without_age=int(x.Age.isna().sum())))
        for name, col in (("Complexity", "cx"), ("HR", "HR"), ("SDNN", "HRV_SDNN"), ("RMSSD", "HRV_RMSSD"),
                          ("pNN50", "HRV_pNN50"), ("DFA alpha1", "HRV_DFA_alpha1")):
            if col not in x or x[col].isna().all():
                continue
            a, p = auc_p(C[col], P[col])
            t2.append(dict(Cohort=c, Metric=name, AUC=1 - a if name == "HR" else a, p=p))
        a, p = auc_p(C.cx, P.cx)
        lo, hi = boot_ci(C.cx.values, P.cx.values, rng)
        t3.append(dict(Cohort=c, n_Control=len(C), n_PD=len(P),
                       Control_median=C.cx.median(), Control_q1=C.cx.quantile(.25), Control_q3=C.cx.quantile(.75),
                       PD_median=P.cx.median(), PD_q1=P.cx.quantile(.25), PD_q3=P.cx.quantile(.75),
                       p=p, AUC=a, CI_lo=lo, CI_hi=hi))
        X = sm.add_constant(pd.DataFrame({"pd": (x.Group == "PD").astype(float), "HR": x.HR}))
        m = sm.OLS(x.cx, X).fit(cov_type="HC3")
        z = sm.OLS((x.cx - x.cx.mean()) / x.cx.std(), X).fit(cov_type="HC3")
        res = x.cx - m.params["HR"] * x.HR
        aa, _ = auc_p(res[x.Group == "Control"], res[x.Group == "PD"])
        t4.append(dict(Cohort=c, AUC=a, p=p, AUC_HR_adjusted=aa, p_HR_adjusted=m.pvalues["pd"],
                       beta_PD_standardized=z.params["pd"], p_beta=z.pvalues["pd"]))
    for name, t in (("Table1_demographics", t1), ("Table2_conventional", t2),
                    ("Table3_primary", t3), ("Table4_hr_adjustment", t4)):
        pd.DataFrame(t).round(4).to_csv(os.path.join(out, name + ".csv"), index=False)
    print(pd.DataFrame(t3).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
