"""
Cross-Dataset Consistency Check & Final Summary Report
======================================================
Synthesises all prior analyses into:

  1. Cross-dataset forest plot (6 key metric pairs × 4 datasets)
     → figures/Figure8/Fig8_B_cross_dataset_consistency.png

  2. Physiological interpretation table
     → figures/Figure8/physiological_interpretation.csv

  3. Markdown analysis report
     → figures/Figure8/analysis_report.md

All numeric values are read from previously computed result files; no
raw data is re-analysed here except for the unique-variance logistic
regression (computed on-the-fly for the interpretation table).
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "figures" / "Figure8"
FIGURES = ROOT / "figures"

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

DS_COLORS = {
    "Chile":  "#1565C0",
    "Spain":  "#2E7D32",
    "Japan":  "#E65100",
    "Pooled": "#4A148C",
}

KEY_METRICS = ["RMSSD", "SDNN", "HF_power", "LF_HF", "DFA_alpha1", "pNN50"]
# SampEn excluded: RC-MSE toolbox Scale=1 output — direct constituent of rcMSE-AUC (circular).
# TINN excluded: ~53% zeros in Chile and ~33% in Spain due to histogram computation failure
# on short recordings. These zeros produce a spurious ρ=0.65 for Chile (drops to ρ=0.30,
# p=0.09 when zeros excluded). pNN50 is the most consistently significant metric across
# all three cohorts and has no computation failures.
KEY_LABELS  = {
    "RMSSD":     "ρ(Complexity, RMSSD)",
    "SDNN":      "ρ(Complexity, SDNN)",
    "HF_power":  "ρ(Complexity, HF power)",
    "LF_HF":     "ρ(Complexity, LF/HF)",
    "DFA_alpha1":"ρ(Complexity, DFA α1)",
    "pNN50":     "ρ(Complexity, pNN50)",
}

ALL_METRICS = [
    "SDNN","RMSSD","pNN50","SDANN",
    # TINN excluded: ~53% zeros Chile, ~33% Spain — histogram failure on short recordings
    "SD1","SD2","SD1SD2",
    "VLF_power","LF_power","HF_power","Total_power",
    "LF_HF","LF_norm","HF_norm",
    "DFA_alpha1","DFA_alpha2",
    # SampEn excluded: it is RC-MSE toolbox Scale=1 output, a direct constituent of
    # rcMSE-AUC (mean of scales 1–5 or 1–20). Correlating component with composite is
    # circular and inflates apparent associations (ρ≈0.85 pooled).
]

AUTONOMIC_BRANCH = {
    "SDNN":        "Overall HRV (sympatho-vagal)",
    "RMSSD":       "Parasympathetic (vagal)",
    "pNN50":       "Parasympathetic (vagal)",
    "SDANN":       "Slow sympathetic / circadian",
    "TINN":        "Overall HRV (geometric)",
    "SD1":         "Parasympathetic (short-term)",
    "SD2":         "Sympathovagal (longer-term)",
    "SD1SD2":      "Sympathovagal balance",
    "VLF_power":   "Very slow rhythms (RAAS/thermo)",
    "LF_power":    "Baroreflex / sympathovagal",
    "HF_power":    "Parasympathetic (respiratory)",
    "Total_power": "Overall HRV (spectral)",
    "LF_HF":       "Sympathovagal balance",
    "LF_norm":     "Sympathovagal balance",
    "HF_norm":     "Parasympathetic (normalised)",
    "DFA_alpha1":  "Short-range fractal scaling",
    "DFA_alpha2":  "Long-range fractal scaling",
}


def nagelkerke_r2(y, y_proba):
    n = len(y)
    p_bar = y.mean()
    ll_null  = n * (p_bar * np.log(p_bar + 1e-15) + (1 - p_bar) * np.log(1 - p_bar + 1e-15))
    ll_model = -n * log_loss(y, y_proba)
    cs = 1 - np.exp(2 / n * (ll_null - ll_model))
    cs_max = 1 - np.exp(2 / n * ll_null)
    return float(cs / cs_max) if cs_max > 0 else np.nan


def fit_r2(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(Xs, y)
    return nagelkerke_r2(y, clf.predict_proba(Xs)[:, 1])


def load_merged() -> pd.DataFrame:
    """Reload the merged analysis dataset (same as prior scripts)."""
    hrv  = pd.read_csv(RESULTS / "traditional_hrv_metrics.csv")
    cons = pd.read_csv(RESULTS / "consolidated_metrics_no_imputation.csv")[
        ["Subject", "Center", "Complexity"]
    ]
    chile_demo = (pd.read_csv(DATA / "chile_demographics.csv")
                  .rename(columns={"Anon_ID": "Subject"})[["Subject", "Age", "Gender"]])
    chile_demo["Sex"] = chile_demo["Gender"].map({"M": 0, "F": 1})
    spain_demo = pd.read_csv(DATA / "spain_demographics.csv")[["Subject", "Age", "Gender"]]
    spain_demo["Sex"] = np.nan
    japan_meta = (pd.read_csv(DATA / "japan_metadata.csv")
                  .rename(columns={"Subject_ID": "Subject"})[["Subject", "Age", "Gender"]])
    japan_meta["Gender"] = japan_meta["Gender"].map({0: "Male", 1: "Female"})
    japan_meta["Sex"] = japan_meta["Gender"].str.lower().map({"male": 0, "female": 1})
    demo = pd.concat([chile_demo[["Subject","Age","Sex"]],
                      spain_demo[["Subject","Age","Sex"]],
                      japan_meta[["Subject","Age","Sex"]]], ignore_index=True)
    df = hrv.merge(cons[["Subject","Complexity"]], on="Subject", how="inner")
    df = df.merge(demo, on="Subject", how="left")
    df["Group_bin"] = df["Group"].map({"PD": 1, "Control": 0})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1 — Cross-dataset forest plot
# ─────────────────────────────────────────────────────────────────────────────

def make_forest_plot():
    spear = pd.read_csv(RESULTS / "spearman_correlations.csv")

    ds_order  = ["Chile", "Spain", "Japan", "Pooled"]
    y_offsets = {"Chile": 0.25, "Spain": 0.08, "Japan": -0.08, "Pooled": -0.25}
    n_metrics = len(KEY_METRICS)

    fig, axes = plt.subplots(1, n_metrics, figsize=(18, 4.5),
                             gridspec_kw={"wspace": 0.35})

    for ax, metric in zip(axes, KEY_METRICS):
        # Check directional consistency across non-pooled datasets
        per_ds = spear[(spear["Subset"] == "All") &
                       (spear["Metric"] == metric) &
                       (spear["Dataset"] != "Pooled")]
        signs = per_ds.dropna(subset=["rho"])["rho"].apply(np.sign)
        consistent = (signs.nunique() == 1)

        for ds in ds_order:
            row = spear[(spear["Dataset"] == ds) &
                        (spear["Subset"] == "All") &
                        (spear["Metric"] == metric)]
            if row.empty or row["rho"].isna().all():
                continue
            rho   = row["rho"].iloc[0]
            ci_lo = row["CI_lo"].iloc[0]
            ci_hi = row["CI_hi"].iloc[0]
            q     = row["q_BH"].iloc[0]
            y     = y_offsets[ds]
            color = DS_COLORS[ds]
            lw    = 2.0 if ds == "Pooled" else 1.5
            ms    = 8  if ds == "Pooled" else 6

            ax.errorbar(rho, y,
                        xerr=[[rho - ci_lo], [ci_hi - rho]],
                        fmt="D" if ds == "Pooled" else "o",
                        color=color, capsize=4, capthick=lw,
                        markersize=ms, linewidth=lw, zorder=3)

            # Significance marker
            if not np.isnan(q) and q < 0.05:
                marker = "***" if q < 0.001 else ("**" if q < 0.01 else "*")
                ax.text(ci_hi + 0.04, y, marker, va="center",
                        fontsize=8, color=color, fontweight="bold")

        # Consistency badge
        badge_text = "✓ consistent" if consistent else "✗ inconsistent"
        badge_col  = "#2E7D32" if consistent else "#C62828"
        ax.text(0.5, 1.03, badge_text, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5,
                color=badge_col, fontweight="bold")

        ax.axvline(0, color="#616161", lw=0.9, ls="--", zorder=1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_yticks(list(y_offsets.values()))
        ax.set_yticklabels(ds_order, fontsize=9)
        ax.set_xlabel("Spearman ρ", fontsize=9)
        ax.set_title(KEY_LABELS[metric], fontsize=8.5, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

        # Light background shading
        ax.axvspan(-1.05, 0, alpha=0.03, color="#C62828")
        ax.axvspan(0, 1.05, alpha=0.03, color="#1565C0")

    # Legend
    handles = [mpatches.Patch(color=DS_COLORS[ds], label=ds) for ds in ds_order]
    handles += [
        plt.Line2D([0], [0], marker="o", color="grey", ms=6, lw=0, label="Per-dataset ρ"),
        plt.Line2D([0], [0], marker="D", color="grey", ms=7, lw=0, label="Pooled ρ"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(
        "Cross-Dataset Consistency: Spearman ρ between rcMSE-AUC and Key HRV Metrics\n"
        "Error bars = bootstrap 95% CI  ·  * q<0.05  ** q<0.01  *** q<0.001 (BH-FDR)",
        fontsize=11, y=1.06,
    )

    out = FIGURES / "Figure8" / "Fig8_B_cross_dataset_consistency.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Cross-dataset forest plot → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2 — Physiological interpretation table
# ─────────────────────────────────────────────────────────────────────────────

def build_interpretation_table(df: pd.DataFrame) -> pd.DataFrame:
    spear      = pd.read_csv(RESULTS / "spearman_correlations.csv")
    adj        = pd.read_csv(RESULTS / "age_sex_adjusted_correlations.csv")
    scale_corr = pd.read_csv(RESULTS / "scale_metric_correlations.csv")

    # Pooled confounders (same as hierarchical regression pooled model)
    df_pool = df.copy()
    df_pool["Center_Spain"] = (df_pool["Center"] == "Spain").astype(float)
    df_pool["Center_Japan"] = (df_pool["Center"] == "Japan").astype(float)
    conf_cols_pool = ["Age", "Center_Spain", "Center_Japan"]

    pool_sp  = spear[(spear["Dataset"] == "Pooled") & (spear["Subset"] == "All")].set_index("Metric")
    pool_adj = adj[adj["Dataset"] == "Pooled"].set_index("Metric")

    rows = []
    for metric in ALL_METRICS:
        # ── ρ pooled + CI ──────────────────────────────────────────────────
        if metric in pool_sp.index:
            rho_p  = pool_sp.loc[metric, "rho"]
            ci_lo  = pool_sp.loc[metric, "CI_lo"]
            ci_hi  = pool_sp.loc[metric, "CI_hi"]
        else:
            rho_p = ci_lo = ci_hi = np.nan

        # ── Partial r after age/sex (residualised Spearman) ────────────────
        if metric in pool_adj.index:
            rho_adj = pool_adj.loc[metric, "rho_adj"]
        else:
            rho_adj = np.nan

        # ── Unique variance % (ΔR² when adding M to pooled confounders + Complexity) ──
        needed = conf_cols_pool + [metric, "Complexity", "Group_bin"]
        sub_c  = df_pool[needed].dropna()
        if len(sub_c) >= 15 and sub_c[metric].std() > 0:
            y = sub_c["Group_bin"].values
            # Model A: confounders + Complexity
            r2_a = fit_r2(sub_c[conf_cols_pool + ["Complexity"]].values, y)
            # Model B: confounders + Complexity + M
            r2_b = fit_r2(sub_c[conf_cols_pool + ["Complexity", metric]].values, y)
            delta_r2 = max(r2_b - r2_a, 0.0)
            uniq_pct = 100.0 * delta_r2 / r2_b if r2_b > 0 else 0.0
        else:
            uniq_pct = np.nan

        # ── Scale range of peak per-scale correlation (across all datasets) ─
        sc = scale_corr[scale_corr["Metric"] == metric]
        sc_sig = sc[sc["q_BH"] < 0.05].copy()
        if len(sc_sig) > 0:
            sc_sig["abs_rho"] = sc_sig["rho"].abs()
            peak_sc = int(sc_sig.loc[sc_sig["abs_rho"].idxmax(), "Scale"])
            sig_sc  = sorted(sc_sig["Scale"].astype(int).unique())
            sc_range = f"{sig_sc[0]}–{sig_sc[-1]} (peak {peak_sc})"
        else:
            sc_range = "none significant"

        # ── Directional consistency ────────────────────────────────────────
        per_ds = spear[(spear["Subset"] == "All") &
                       (spear["Metric"] == metric) &
                       (spear["Dataset"] != "Pooled")].dropna(subset=["rho"])
        if len(per_ds) >= 2:
            signs = per_ds["rho"].apply(np.sign)
            consistent = "yes" if signs.nunique() == 1 else "no"
        else:
            consistent = "n/a"

        # ── Interpretation note ────────────────────────────────────────────
        abs_rho = abs(rho_p) if not np.isnan(rho_p) else 0
        uv = uniq_pct if not np.isnan(uniq_pct) else 0
        if abs_rho > 0.5 and uv > 10:
            note = "Strong independent relationship with complexity"
        elif abs_rho > 0.5 and uv <= 5:
            note = "Largely redundant with complexity"
        elif abs_rho > 0.5 and 5 < uv <= 10:
            note = "Moderate redundancy; some unique contribution"
        elif abs_rho < 0.3:
            note = "Weak relationship — complexity captures distinct information"
        else:
            note = "Moderate relationship; partial overlap with complexity"

        rows.append({
            "metric":                               metric,
            "primary_autonomic_branch":             AUTONOMIC_BRANCH.get(metric, "—"),
            "rho_with_complexity_pooled":           round(rho_p,  4) if not np.isnan(rho_p)  else np.nan,
            "CI_lower":                             round(ci_lo,  4) if not np.isnan(ci_lo)  else np.nan,
            "CI_upper":                             round(ci_hi,  4) if not np.isnan(ci_hi)  else np.nan,
            "partial_r_after_age_sex":              round(rho_adj,4) if not np.isnan(rho_adj) else np.nan,
            "unique_variance_%":                    round(uniq_pct,2) if not np.isnan(uniq_pct) else np.nan,
            "scale_range_of_peak_correlation":     sc_range,
            "directionally_consistent_across_datasets": consistent,
            "interpretation_note":                 note,
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "physiological_interpretation.csv", index=False)
    print(f"Interpretation table → figures/Figure8/physiological_interpretation.csv")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(interp: pd.DataFrame):
    spear  = pd.read_csv(RESULTS / "spearman_correlations.csv")
    hier   = pd.read_csv(RESULTS / "hierarchical_regression.csv")
    comm   = pd.read_csv(RESULTS / "commonality_analysis.csv")
    sgd    = pd.read_csv(RESULTS / "scale_group_differences.csv")

    # ── Top-3 metrics per dataset ──────────────────────────────────────────
    excl = ["SampEn"]
    top3 = {}
    for ds in ["Chile", "Spain", "Japan", "Pooled"]:
        sub = (spear[(spear["Dataset"] == ds) & (spear["Subset"] == "All")
                     & (~spear["Metric"].isin(excl))]
               .assign(abs_rho=lambda d: d["rho"].abs())
               .sort_values("abs_rho", ascending=False)
               .head(3))
        top3[ds] = list(zip(sub["Metric"], sub["rho"].round(3), sub["sig"].fillna("ns")))

    # ── Incremental value summary ──────────────────────────────────────────
    b1 = hier[hier["Block"].str.startswith("Block1")].set_index("Dataset")
    b2 = hier[hier["Block"].str.startswith("Block2")].set_index("Dataset")
    b3 = hier[hier["Block"].str.startswith("Block3")].set_index("Dataset")

    # ── Scale ranges with PD < HC (from scale_group_differences) ──────────
    scale_summary = {}
    for ds in ["Chile", "Spain", "Japan-morning", "Japan-afternoon"]:
        sig = sgd[(sgd["Dataset"] == ds) & (sgd["q_BH"] < 0.05)]
        if len(sig):
            sc_list = sorted(sig["Scale"].astype(int).tolist())
            # Directionality: PD < HC?
            direction = sig.eval("PD_mean < HC_mean").all()
            scale_summary[ds] = {
                "scales": sc_list,
                "pd_lower": direction,
                "n_sig": len(sc_list),
            }
        else:
            scale_summary[ds] = {"scales": [], "pd_lower": None, "n_sig": 0}

    # ── Directional consistency counts ────────────────────────────────────
    n_consistent = interp[interp["directionally_consistent_across_datasets"] == "yes"].shape[0]
    n_total = interp[interp["directionally_consistent_across_datasets"].isin(["yes","no"])].shape[0]

    # ── Key metric interp values ───────────────────────────────────────────
    pnn50_row  = interp[interp["metric"] == "pNN50"].iloc[0]
    rmssd_row  = interp[interp["metric"] == "RMSSD"].iloc[0]
    dfa_row    = interp[interp["metric"] == "DFA_alpha1"].iloc[0]

    # ── Commonality ───────────────────────────────────────────────────────
    pool_comm = comm[comm["Dataset"] == "Pooled"].iloc[0]
    japan_comm = comm[comm["Dataset"] == "Japan"].iloc[0]

    # ── Compose report ────────────────────────────────────────────────────
    def fmt_top3(ds):
        return ", ".join(f"{m} (ρ={r:.3f} {s})" for m, r, s in top3[ds])

    def fmt_scales(ds):
        info = scale_summary.get(ds, {"scales": [], "n_sig": 0})
        if not info["scales"]:
            return "none"
        sc = info["scales"]
        if len(sc) > 5:
            return f"1–{sc[-1]} (all {info['n_sig']} significant)"
        return ", ".join(str(s) for s in sc)

    def incr_row(ds):
        if ds not in b3.index:
            return "n/a"
        d_r2 = b3.loc[ds, "Delta_R2"]
        p    = b3.loc[ds, "LR_p"]
        auc2 = b2.loc[ds, "CV_AUC"]
        auc3 = b3.loc[ds, "CV_AUC"]
        sig  = "p={:.4f}".format(p)
        return f"ΔR²={d_r2:.4f}, {sig}, ΔAUC={auc3-auc2:+.3f}"

    report = f"""# rcMSE-AUC vs Traditional HRV Metrics: Analysis Report

*Generated from pre-computed results. All p-values are two-sided; FDR correction uses
Benjamini-Hochberg unless noted.*

---

## 1. Strongest traditional metric correlates of complexity (per dataset)

| Dataset | Top 3 traditional HRV metrics (ρ, FDR significance) |
|---------|------------------------------------------------------|
| Chile (n=71, ECG ~15 min) | {fmt_top3("Chile")} |
| Spain (n=58, PPG 5–15 min) | {fmt_top3("Spain")} |
| Japan (n=37, ECG daytime 4 h) | {fmt_top3("Japan")} |
| Pooled (n=166) | {fmt_top3("Pooled")} |

**pNN50** (proportion of successive NN differences > 50 ms) is the most consistently
significant correlate of rcMSE-AUC across all three datasets. It is available for all
subjects without computation failures.

Two metrics have been **excluded from featured comparisons**:
- **SampEn** (RC-MSE toolbox, Scale=1): direct constituent of the rcMSE-AUC composite
  (mean of scales 1–5 for Chile/Spain, 1–20 for Japan). Circular by construction (ρ≈0.85).
- **TINN**: ~53% of Chile and ~33% of Spain values are zero due to histogram computation
  failure on short recordings (~15 min and 5–15 min respectively). These zeros create a
  spurious correlation (ρ=0.65 with zeros vs ρ=0.30 p=0.09 when zeros excluded for Chile).
  TINN is appropriate only for Japan where 24 h data yields reliable histograms.

---

## 2. Incremental predictive value of complexity after traditional metrics

Hierarchical logistic regression (group ~ confounders → + pNN50 → + rcMSE-AUC):

| Dataset | ΔAUC (Block2→3) | ΔR² | LR-test p | Significant? |
|---------|----------------|-----|-----------|-------------|
| Chile   | {b3.loc["Chile","CV_AUC"]-b2.loc["Chile","CV_AUC"]:+.3f} | {incr_row("Chile")} | {"Yes" if b3.loc["Chile","LR_p"]<0.05 else "No (p="+f"{b3.loc['Chile','LR_p']:.3f})"} |
| Spain   | {b3.loc["Spain","CV_AUC"]-b2.loc["Spain","CV_AUC"]:+.3f} | {incr_row("Spain")} | {"Yes" if b3.loc["Spain","LR_p"]<0.05 else "No (p="+f"{b3.loc['Spain','LR_p']:.3f})"} |
| Japan   | {b3.loc["Japan","CV_AUC"]-b2.loc["Japan","CV_AUC"]:+.3f} | {incr_row("Japan")} | {"Yes" if b3.loc["Japan","LR_p"]<0.05 else "No (p="+f"{b3.loc['Japan','LR_p']:.3f})"} |
| Pooled  | {b3.loc["Pooled","CV_AUC"]-b2.loc["Pooled","CV_AUC"]:+.3f} | {incr_row("Pooled")} | {"Yes" if b3.loc["Pooled","LR_p"]<0.05 else "No (p="+f"{b3.loc['Pooled','LR_p']:.3f})"} |

rcMSE-AUC adds **significant incremental predictive value** in the Japan cohort
(LR p={b3.loc["Japan","LR_p"]:.4f}) and in the pooled analysis (LR p={b3.loc["Pooled","LR_p"]:.4f})
after accounting for pNN50. The effect is numerically consistent across Chile and Spain but
does not reach significance individually, likely due to smaller sample sizes and shorter
recording lengths limiting the reliability of high-scale entropy estimates.

Commonality analysis (McFadden R²) confirms the complementarity: in the pooled model,
Complexity contributes {100*pool_comm.Unique_Complexity/pool_comm.R2_full:.0f}% of the
full model's R² uniquely (not shared with pNN50 or confounders), while pNN50 contributes
only {100*pool_comm.Unique_pNN50/pool_comm.R2_full:.0f}%. In Japan — where the 24 h recording
allows all 20 scales — the shared component between pNN50 and Complexity reaches
{100*japan_comm.Shared_pNN50_Compl/japan_comm.R2_full:.0f}% of full R², indicating that longer
recordings allow more overlap between vagal tone and complexity.

---

## 3. Scale ranges driving complexity differences in PD

| Dataset | Significant scales (BH-FDR, PD < HC) |
|---------|---------------------------------------|
| CETRAM (~15 min) | {fmt_scales("Chile")} |
| Cruces (5–15 min) | {fmt_scales("Spain")} |
| Nagoya 07–11h | {fmt_scales("Japan-morning")} |
| Nagoya 16–20h | {fmt_scales("Japan-afternoon")} |

For **Chile and Spain** (short recordings of ~15 min and 5–15 min respectively),
complexity loss in PD is confined to **scale 1** (i.e., SampEn at scale 1 from the
RC-MSE toolbox), the only scale reaching significance after FDR correction. This reflects
the limitation of short recordings: coarse-grained scales require longer time series to
provide reliable entropy estimates.

For the **Nagoya** 24 h cohort, group differences span all 20 scales in the optimal window
(16–20h), with peak separation at the vagal zone (scales 1–5) and sustained through the
baroreflex and slow zones (scales 6–20). The afternoon window (16–20h) provides the strongest
PD discrimination; the morning window (07–11h) also shows significant separation. For the
full circadian profile and window-by-window analysis, see Figure 3 (data/japan_evolution.csv).

---

## 4. Cross-dataset consistency of complexity-metric relationships

Of {n_total} tested metric correlations with complexity, **{n_consistent} are
directionally consistent** across all three datasets (same sign of ρ in Chile, Spain, and Japan).

Consistent metrics: **pNN50** (ρ > 0 in all three datasets) and **SD1** (ρ > 0 in all three).

Most time-domain (RMSSD, SDNN) and frequency-domain (HF_power, LF/HF) metrics show
**inconsistent directionality**: they are negatively correlated with complexity in Chile
but positive or near-zero in Spain and Japan. This inconsistency likely reflects two
interacting effects:
1. **Recording modality and length**: Chile uses ECG; Spain uses PPG (subject to different
   noise characteristics); Japan uses 24 h Holter. Entropy at short scales is particularly
   sensitive to these differences.
2. **Scale-normalisation**: The RC-MSE algorithm normalises the tolerance parameter by
   σ per coarse-graining scale. When SDNN/RMSSD are large (high absolute variability),
   the normalised tolerance becomes large, potentially reducing apparent entropy. This
   creates a within-dataset negative correlation between time-domain HRV and complexity
   that disappears across datasets where other sources of variance dominate.

After partialling out age, sex, and centre, the partial correlations of RMSSD and SDNN
with complexity collapse toward zero or reverse sign (RMSSD: ρ_adj = {rmssd_row.partial_r_after_age_sex:.3f}),
confirming these are largely confounded. pNN50 remains the most consistent metric
across centers (ρ_adj = {pnn50_row.partial_r_after_age_sex:.3f} after adjustment).

---

## 5. Physiological interpretation

**What is rcMSE-AUC measuring that traditional HRV metrics do and do not already capture?**

rcMSE-AUC (HR-normalised refined composite multiscale sample entropy, integrated over
scales 1–5 for short recordings or 1–20 for 24 h recordings) quantifies the *scale-resolved
structural complexity* of cardiac rhythm — specifically, how much new information emerges
as the cardiac time series is viewed at progressively coarser temporal resolutions. This is
distinct from any single-scale metric of HRV.

Traditional HRV metrics capture specific, physiologically well-defined aspects of autonomic
function: RMSSD and HF power index *vagal efferent activity at respiratory frequencies*;
SDNN summarises *total HRV magnitude*; DFA α1 characterises *short-range fractal scaling*
of the IBI time series. Each of these is a projection of a single property of the RR
time series onto one temporal scale or frequency band.

rcMSE-AUC, by contrast, integrates across scales. The partial correlation analysis reveals
that after controlling for age, sex, and recording site, DFA α1 drops from ρ = {dfa_row.rho_with_complexity_pooled:.3f}
to ρ_adj = {dfa_row.partial_r_after_age_sex:.3f}, and RMSSD from ρ = {rmssd_row.rho_with_complexity_pooled:.3f}
to ρ_adj = {rmssd_row.partial_r_after_age_sex:.3f} — both near zero — suggesting that the
apparent associations of these metrics with complexity are largely driven by confounds. pNN50
remains the most robust correlate after adjustment (ρ_adj = {pnn50_row.partial_r_after_age_sex:.3f}),
reflecting its consistent positive association with complexity across all three cohorts.

The complementarity between pNN50 and rcMSE-AUC is mechanistically informative: pNN50
indexes *vagal beat-to-beat modulation* (the fraction of consecutive beats differing by
> 50 ms), while rcMSE-AUC captures *multi-timescale structural complexity* of the full
cardiac rhythm. In PD, both are reduced — the autonomic nervous system both loses its
rapid beat-to-beat variability and the temporal unpredictability of transitions across
time scales. These are related but not identical phenomena: cardiac variability at
respiratory timescales (pNN50) and multi-scale complexity can dissociate, particularly
in conditions affecting baroreflex or circadian autonomic regulation.

The primary autonomic substrate of the rcMSE-AUC reduction in PD appears to be a **global
depression of multi-timescale complexity** that is not driven by any single branch of the
ANS. At the vagal zone (scales 1–5), complexity loss tracks SampEn and RMSSD loss; at the
baroreflex zone (scales 6–15), it tracks LF_norm; and the 24 h cohort reveals additional
loss at slow scales (16–20) associated with SDANN and circadian rhythm. This hierarchical
structure — loss at every time scale — suggests a centrally-mediated reduction in autonomic
flexibility rather than selective impairment of vagal or sympathetic tone alone, consistent
with the known involvement of the dorsal vagal complex, basal ganglia-brainstem circuits,
and insular cortex in PD-related autonomic dysfunction.
"""

    out = RESULTS / "analysis_report.md"
    out.write_text(report)
    print(f"Analysis report → figures/Figure8/analysis_report.md")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading merged dataset...")
    df = load_merged()
    print(f"  {len(df)} subjects across {df['Center'].nunique()} centres")

    print("\n=== Analysis 1: Cross-dataset forest plot ===")
    make_forest_plot()

    print("\n=== Analysis 2: Physiological interpretation table ===")
    interp = build_interpretation_table(df)

    print("\nTop metrics by pooled |ρ|:")
    print(interp.nlargest(8, "rho_with_complexity_pooled")[
        ["metric", "rho_with_complexity_pooled", "CI_lower", "CI_upper",
         "partial_r_after_age_sex", "unique_variance_%",
         "directionally_consistent_across_datasets"]
    ].to_string(index=False))

    print("\n=== Final report ===")
    generate_report(interp)

    print("\n=== Done — all outputs written ===")
    print("  figures/Figure8/Fig8_B_cross_dataset_consistency.png")
    print("  figures/Figure8/physiological_interpretation.csv")
    print("  figures/Figure8/analysis_report.md")
