"""
Cross-Dataset Consistency Check
================================
Synthesises all prior analyses into:

  1. Cross-dataset forest plot (6 key metric pairs × 4 datasets)
     → figures/Figure7/Fig7_B_cross_dataset_consistency.png

  2. Physiological interpretation table
     → figures/Figure7/physiological_interpretation.csv

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
RESULTS = ROOT / "figures" / "Figure7"
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

    out = FIGURES / "Figure7" / "Fig7_B_cross_dataset_consistency.png"
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
    print(f"Interpretation table → figures/Figure7/physiological_interpretation.csv")
    return out




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

    print("\n=== Done — all outputs written ===")
    print("  figures/Figure7/Fig7_B_cross_dataset_consistency.png")
    print("  figures/Figure7/physiological_interpretation.csv")
