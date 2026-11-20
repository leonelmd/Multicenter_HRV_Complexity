#!/usr/bin/env python3
"""
Supplementary Figs. S6 and S7: cross-cohort transfer of the group contrast.

Replaces the classification panels of the old Figure 5 (generate_figure5.py) and the
random-forest generalization matrix (generate_appendix.py), which used the withdrawn
HR-normalized index, an outcome-selected Nagoya window for the deep-learning reference,
and flipped any held-out AUC below 0.5 to 1 - AUC.

Design (fixed before running; no tuning)
----------------------------------------
* Complexity index: canonical nAUC (trapz over scales / n), unnormalized; scales 1-5
  for CETRAM and Cruces, 1-20 for Nagoya 18:00-22:00. Same definition as
  verify_manuscript_claims.py and Table 2 of the manuscript.
* Heart rate enters as a covariate, as in the primary analysis.
* Three prespecified feature sets:
      HR            heart rate only
      Complexity    complexity index + HR
      Conventional  SDNN + RMSSD + HR
* Features are z-scored within each cohort, label-blind (the held-out cohort is
  standardized on its own feature distribution, never on its labels). This removes
  the cohort offset created by different scale ranges, devices and recording lengths.
* Model: L2-penalized logistic regression (C = 1, balanced class weights, unpenalized
  intercept), fitted by Newton iterations. No hyperparameter search.
* Orientation: AUC = P(score_PD > score_Control). AUCs below 0.5 are reported as such.
* Uncertainty: stratified bootstrap of held-out participants (4000 replicates), with
  the fitted model held fixed; the complexity-vs-HR difference uses paired replicates.

S6: leave-one-cohort-out (train on two cohorts, test on the third).
S7: single-cohort transfer matrix (train on one cohort, test on each other one).
    The HR-only model is not drawn there: with a single feature its ranking of a test
    cohort is the same whichever cohort it was trained on (its AUCs are in the CSV).

Dependencies: numpy, pandas, matplotlib only.
Run:  python3 scripts/figS6_S7_cross_cohort.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as fs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures", "FigS6_S7_cross_cohort")
COHORTS = ["CETRAM", "Cruces", "Nagoya"]
MODELS = {
    "HR": ["HR"],
    "Complexity": ["idx", "HR"],
    "Conventional": ["SDNN", "RMSSD", "HR"],
}
MODEL_LABEL = {"HR": "Heart rate only", "Complexity": "Complexity + HR",
               "Conventional": "SDNN + RMSSD + HR"}
MODEL_COLOR = {"HR": fs.MUTED, "Complexity": fs.ACCENT, "Conventional": "#55A868"}
NBOOT = 4000
RNG = np.random.default_rng(20261102)


# ── data ──────────────────────────────────────────────────────────────────────
def nauc(c):
    c = np.asarray(c, float)
    if len(c) == 0 or not np.isfinite(c).all():
        return np.nan
    return float(np.trapezoid(c, np.arange(1, len(c) + 1)) / len(c))


def index_table(fname, lo, hi):
    d = pd.read_csv(os.path.join(D, fname))
    d = d[(d.Scales >= lo) & (d.Scales <= hi)].sort_values("Scales")
    return d.groupby("Subject").MSE.apply(nauc).rename("idx")


def load():
    rows = []
    for coh, mfile, msefile, hi in [("CETRAM", "chile_metrics.csv", "chile_mse.csv", 5),
                                     ("Cruces", "spain_metrics.csv", "spain_mse.csv", 5)]:
        m = pd.read_csv(os.path.join(D, mfile))
        m = m[m.Group.isin(["Control", "PD"])].copy()
        m["HR"] = 60000.0 / m.HRV_MeanNN
        m = m.rename(columns={"HRV_SDNN": "SDNN", "HRV_RMSSD": "RMSSD"})
        m["idx"] = m.Subject.map(index_table(msefile, 1, hi))
        m["Cohort"] = coh
        rows.append(m[["Cohort", "Subject", "Group", "idx", "HR", "SDNN", "RMSSD"]])
    j = pd.read_csv(os.path.join(D, "japan_window_features.csv"))
    j = j.rename(columns={"HRV_SDNN": "SDNN", "HRV_RMSSD": "RMSSD"})
    j["idx"] = j.Subject.map(index_table("japan_window_mse.csv", 1, 20))
    j["Cohort"] = "Nagoya"
    rows.append(j[["Cohort", "Subject", "Group", "idx", "HR", "SDNN", "RMSSD"]])
    df = pd.concat(rows, ignore_index=True).dropna()
    df["y"] = (df.Group == "PD").astype(int)
    feats = ["idx", "HR", "SDNN", "RMSSD"]
    for c in COHORTS:                       # label-blind, within-cohort z-score
        m = df.Cohort == c
        df.loc[m, feats] = (df.loc[m, feats] - df.loc[m, feats].mean()) / df.loc[m, feats].std(ddof=0)
    return df


# ── model ─────────────────────────────────────────────────────────────────────
def fit_logreg(X, y, C=1.0, iters=50):
    """L2 logistic regression, balanced weights, unpenalized intercept (sklearn objective)."""
    n = len(y)
    w_s = np.where(y == 1, n / (2 * y.sum()), n / (2 * (n - y.sum())))
    Xa = np.column_stack([np.ones(n), X])
    beta = np.zeros(Xa.shape[1])
    P = np.eye(Xa.shape[1]) / C
    P[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xa @ beta))
        g = Xa.T @ (w_s * (p - y)) + P @ beta
        H = (Xa * (w_s * p * (1 - p))[:, None]).T @ Xa + P
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def predict(beta, X):
    return 1 / (1 + np.exp(-(beta[0] + X @ beta[1:])))


def auc(y, s):
    """P(score_PD > score_Control), ties counted half."""
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def roc(y, s):
    order = np.argsort(-s, kind="mergesort")
    y = np.asarray(y)[order]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / (1 - y).sum()])
    return fpr, tpr


def boot_idx(y):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    return [np.concatenate([RNG.choice(i1, len(i1)), RNG.choice(i0, len(i0))])
            for _ in range(NBOOT)]


def run(df, train, test, boots):
    tr, te = df[df.Cohort.isin(train)], df[df.Cohort == test]
    out = {}
    for name, cols in MODELS.items():
        b = fit_logreg(tr[cols].to_numpy(), tr.y.to_numpy())
        out[name] = predict(b, te[cols].to_numpy())
    y = te.y.to_numpy()
    res = {"y": y, "scores": out}
    bs = {k: np.array([auc(y[i], v[i]) for i in boots]) for k, v in out.items()}
    for k, v in out.items():
        res[k] = (auc(y, v), *np.percentile(bs[k], [2.5, 97.5]))
    d = bs["Complexity"] - bs["HR"]
    res["delta"] = (res["Complexity"][0] - res["HR"][0], *np.percentile(d, [2.5, 97.5]))
    return res


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    fs.apply()
    df = load()
    print(df.groupby(["Cohort", "Group"]).size().unstack())
    boots = {c: boot_idx(df[df.Cohort == c].y.to_numpy()) for c in COHORTS}

    # within-cohort reference: AUC of the z-scored index itself
    rows = []
    loco = {}
    for c in COHORTS:
        r = run(df, [k for k in COHORTS if k != c], c, boots[c])
        loco[c] = r
        for k in MODELS:
            rows.append(dict(design="LOCO", train="+".join(k2 for k2 in COHORTS if k2 != c),
                             test=c, model=k, auc=r[k][0], lo=r[k][1], hi=r[k][2]))
        rows.append(dict(design="LOCO", train="+".join(k2 for k2 in COHORTS if k2 != c),
                         test=c, model="Complexity-HR", auc=r["delta"][0],
                         lo=r["delta"][1], hi=r["delta"][2]))
    pair = {}
    for tr in COHORTS:
        for te in COHORTS:
            if tr == te:
                continue
            r = run(df, [tr], te, boots[te])
            pair[(tr, te)] = r
            for k in MODELS:
                rows.append(dict(design="single", train=tr, test=te, model=k,
                                 auc=r[k][0], lo=r[k][1], hi=r[k][2]))
            rows.append(dict(design="single", train=tr, test=te, model="Complexity-HR",
                             auc=r["delta"][0], lo=r["delta"][1], hi=r["delta"][2]))
    stats = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    stats.round(4).to_csv(os.path.join(OUT, "FigS6_S7_stats.csv"), index=False)
    print(stats.round(3).to_string(index=False))

    # ── S6: LOCO ──────────────────────────────────────────────────────────────
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1),
                               gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.35))
    w = 0.26
    for i, k in enumerate(MODELS):
        x = np.arange(3) + (i - 1) * w
        v = np.array([loco[c][k] for c in COHORTS])
        a.bar(x, v[:, 0], w * 0.92, color=MODEL_COLOR[k], label=MODEL_LABEL[k])
        a.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
                   fmt="none", ecolor="0.25", elinewidth=0.8, capsize=2)
    a.axhline(0.5, color="0.4", lw=0.8, ls="--")
    a.set_xticks(range(3))
    a.set_xticklabels([f"{c}\n(n={len(loco[c]['y'])})" for c in COHORTS])
    a.set_ylim(0.2, 1.16)
    a.set_yticks(np.arange(0.2, 1.01, 0.2))
    a.set_ylabel("AUC in held-out cohort")
    a.set_xlabel("Held-out cohort")
    a.legend(loc="upper left", fontsize=7.5, ncol=2, columnspacing=1.0)
    fs.panel(a, "a")
    for c in COHORTS:
        fpr, tpr = roc(loco[c]["y"], loco[c]["scores"]["Complexity"])
        b.plot(fpr, tpr, color=fs.COHORT[c], ls=fs.COHORT_LS[c], lw=1.4, drawstyle="steps-post",
               label=f"{c} ({loco[c]['Complexity'][0]:.2f})")
    b.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
    b.set_xlabel("False-positive rate"); b.set_ylabel("True-positive rate")
    b.set_aspect("equal"); b.set_xlim(0, 1); b.set_ylim(0, 1.01)
    b.legend(loc="lower right", fontsize=7.5, title="Complexity + HR", title_fontsize=7.5)
    fs.panel(b, "b")
    fs.save(fig, OUT, "FigS6_loco")

    # ── S7: single-cohort transfer ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(wspace=0.45))
    for ax, k, letter in zip(axes, ["Complexity", "Conventional"], "ab"):
        M = np.full((3, 3), np.nan)
        for (tr, te), r in pair.items():
            M[COHORTS.index(tr), COHORTS.index(te)] = r[k][0]
        for c in COHORTS:
            M[COHORTS.index(c), COHORTS.index(c)] = np.nan
        im = ax.imshow(M, cmap="RdBu_r", vmin=0.2, vmax=0.8)
        for i in range(3):
            for j in range(3):
                if i == j:
                    ax.text(j, i, "–", ha="center", va="center", color="0.5")
                    continue
                r = pair[(COHORTS[i], COHORTS[j])][k]
                ax.text(j, i, f"{r[0]:.2f}\n[{r[1]:.2f}–{r[2]:.2f}]", ha="center",
                        va="center", fontsize=7)
        ax.set_xticks(range(3)); ax.set_xticklabels(COHORTS)
        ax.set_yticks(range(3)); ax.set_yticklabels(COHORTS)
        ax.set_xlabel("Test cohort"); ax.set_ylabel("Training cohort")
        ax.grid(False)
        ax.set_title(MODEL_LABEL[k])
        fs.panel(ax, letter, dx=-0.32)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cb.set_label("AUC")
    fs.save(fig, OUT, "FigS7_transfer")
    print("saved FigureS6_loco and FigureS7_transfer")


if __name__ == "__main__":
    main()
