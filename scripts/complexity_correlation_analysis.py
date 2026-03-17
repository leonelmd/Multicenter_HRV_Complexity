"""
Complexity vs Traditional HRV Metrics — Correlation Analysis
=============================================================
Two analyses:
  1. Spearman ρ between rcMSE-AUC (Complexity) and each traditional HRV metric,
     per dataset and pooled, for All / PD-only / HC-only subsets.
     Bootstrap 95% CI, BH-FDR correction. → figures/Figure7/spearman_correlations.csv
     Correlation heatmap → figures/Figure7/Fig7_A_correlation_heatmap.png

  2. Partial correlations:
     (a) Complexity ~ Group (PD=1/HC=0), controlling for each HRV metric
     (b) Complexity ~ each HRV metric, controlling for Age + Sex
     → figures/Figure7/partial_correlations.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "figures" / "Figure7"
FIGURES = ROOT / "figures"

RNG = np.random.default_rng(42)
N_BOOT = 1000

# ---------------------------------------------------------------------------
# Metrics analysed
# ---------------------------------------------------------------------------
METRIC_COLS = [
    "SDNN", "RMSSD", "pNN50", "SDANN",
    # TINN excluded: ~53% zeros Chile, ~33% Spain — spurious correlations
    "SD1", "SD2", "SD1SD2",
    "VLF_power", "LF_power", "HF_power", "Total_power",
    "LF_HF", "LF_norm", "HF_norm",
    "DFA_alpha1", "DFA_alpha2",
    # NOTE: SampEn (RC-MSE toolbox, Scale=1) is excluded here because it is a direct
    # constituent of the rcMSE-AUC composite score (mean of scales 1–5 or 1–20).
    # Correlating a component with its sum is circular and not scientifically informative.
]

CENTERS = ["Chile", "Spain", "Japan"]

# ---------------------------------------------------------------------------
# BH-FDR (manual, scipy-version-agnostic)
# ---------------------------------------------------------------------------

def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = p[order] * n / (np.arange(1, n + 1))
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return np.clip(q, 0, 1)


def stars(q: float) -> str:
    if np.isnan(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Spearman ρ + bootstrap 95% CI
# ---------------------------------------------------------------------------

def spearman_boot(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT):
    """Return rho, p, ci_lo, ci_hi. NaN if <5 complete pairs."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan
    rho, p = stats.spearmanr(x, y)
    boot_rhos = []
    n = len(x)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        boot_rhos.append(r)
    ci_lo, ci_hi = np.nanpercentile(boot_rhos, [2.5, 97.5])
    return float(rho), float(p), float(ci_lo), float(ci_hi)


# ---------------------------------------------------------------------------
# Data loading & merging
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    hrv = pd.read_csv(RESULTS / "traditional_hrv_metrics.csv")

    # Complexity from consolidated file
    cons = pd.read_csv(RESULTS / "consolidated_metrics_no_imputation.csv")[
        ["Subject", "Center", "Complexity"]
    ]

    # Demographics (Age, Sex/Gender)
    chile_demo = pd.read_csv(DATA / "chile_demographics.csv")[["Anon_ID", "Age", "Gender"]]
    chile_demo = chile_demo.rename(columns={"Anon_ID": "Subject"})

    spain_demo = pd.read_csv(DATA / "spain_demographics.csv")[["Subject", "Age", "Gender"]]

    japan_meta = pd.read_csv(DATA / "japan_metadata.csv")[["Subject_ID", "Gender", "Age"]]
    japan_meta = japan_meta.rename(columns={"Subject_ID": "Subject"})
    # Japan Gender: 0=Male, 1=Female — recode to "Male"/"Female" for uniformity
    japan_meta["Gender"] = japan_meta["Gender"].map({0: "Male", 1: "Female"})

    demo = pd.concat([chile_demo, spain_demo, japan_meta], ignore_index=True)
    # Numeric sex: Male=0, Female=1
    demo["Sex"] = demo["Gender"].str.strip().str.lower().map(
        lambda g: 1 if g == "female" else (0 if g == "male" else np.nan)
    )
    demo = demo[["Subject", "Age", "Sex"]]

    # Merge everything on Subject
    df = hrv.merge(cons[["Subject", "Complexity"]], on="Subject", how="inner")
    df = df.merge(demo, on="Subject", how="left")

    # Binary group
    df["Group_bin"] = df["Group"].map({"PD": 1, "Control": 0})

    print(f"Merged dataset: {len(df)} subjects")
    print(df.groupby(["Center", "Group"]).size().to_string())
    return df


# ---------------------------------------------------------------------------
# Analysis 1 — Spearman correlations
# ---------------------------------------------------------------------------

def run_spearman(df: pd.DataFrame) -> pd.DataFrame:
    subsets = {"All": df, "PD": df[df["Group"] == "PD"], "HC": df[df["Group"] == "Control"]}
    datasets = {c: df[df["Center"] == c] for c in CENTERS}
    datasets["Pooled"] = df

    rows = []
    for ds_name, ds_df in datasets.items():
        for subset_name, _ in subsets.items():
            if subset_name == "All":
                sub = ds_df
            elif subset_name == "PD":
                sub = ds_df[ds_df["Group"] == "PD"]
            else:
                sub = ds_df[ds_df["Group"] == "Control"]

            complexity = sub["Complexity"].values
            p_raw = []
            metric_results = []
            for m in METRIC_COLS:
                rho, p, ci_lo, ci_hi = spearman_boot(sub[m].values, complexity)
                p_raw.append(p if not np.isnan(p) else np.nan)
                metric_results.append((m, rho, p, ci_lo, ci_hi))

            # BH FDR across metrics within this (dataset, subset)
            valid = [(i, p) for i, p in enumerate(p_raw) if not np.isnan(p)]
            if valid:
                idxs, ps = zip(*valid)
                qs = bh_fdr(np.array(ps))
                q_map = dict(zip(idxs, qs))
            else:
                q_map = {}

            for i, (m, rho, p, ci_lo, ci_hi) in enumerate(metric_results):
                q = q_map.get(i, np.nan)
                rows.append({
                    "Dataset": ds_name,
                    "Subset": subset_name,
                    "Metric": m,
                    "n": int(sub[m].dropna().shape[0]),
                    "rho": rho,
                    "p": p,
                    "q_BH": q,
                    "sig": stars(q),
                    "CI_lo": ci_lo,
                    "CI_hi": ci_hi,
                })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "spearman_correlations.csv", index=False)
    print(f"\nSpearman results: {len(out)} rows → figures/Figure7/spearman_correlations.csv")
    return out


# ---------------------------------------------------------------------------
# Analysis 1b — Heatmap
# ---------------------------------------------------------------------------

def make_heatmap(spear: pd.DataFrame):
    # Columns: Dataset × Subset, Rows: Metric
    ds_order = CENTERS + ["Pooled"]
    subset_order = ["All", "PD", "HC"]
    col_labels = [f"{d}\n{s}" for d in ds_order for s in subset_order]

    # Build rho matrix and annotation matrix
    rho_mat = np.full((len(METRIC_COLS), len(col_labels)), np.nan)
    ann_mat = [["" for _ in col_labels] for _ in METRIC_COLS]

    col_idx = {f"{d}_{s}": i for i, (d, s) in
               enumerate([(d, s) for d in ds_order for s in subset_order])}

    for _, row in spear.iterrows():
        r = METRIC_COLS.index(row["Metric"]) if row["Metric"] in METRIC_COLS else None
        c = col_idx.get(f"{row['Dataset']}_{row['Subset']}")
        if r is None or c is None:
            continue
        if not np.isnan(row["rho"]):
            rho_mat[r, c] = row["rho"]
            ann_mat[r][c] = row["sig"]

    rho_df = pd.DataFrame(rho_mat, index=METRIC_COLS, columns=col_labels)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        rho_df, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="lightgrey",
        annot=np.array(ann_mat), fmt="s",
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
    )
    ax.set_title(
        "Spearman ρ: rcMSE-AUC vs Traditional HRV Metrics\n"
        "(* q<0.05, ** q<0.01, *** q<0.001, BH-FDR corrected)",
        fontsize=12, pad=12,
    )
    ax.set_xlabel("Dataset × Subject Subset")
    ax.set_ylabel("HRV Metric")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=9)

    # Vertical dividers between datasets
    for i in range(1, len(ds_order)):
        ax.axvline(i * len(subset_order), color="black", lw=1.5)

    plt.tight_layout()
    out = FIGURES / "Figure7" / "Fig7_A_correlation_heatmap.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Correlation heatmap → {out}")


# ---------------------------------------------------------------------------
# Analysis 2 — Partial correlations
# ---------------------------------------------------------------------------

def run_partial(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    datasets = {c: df[df["Center"] == c] for c in CENTERS}
    datasets["Pooled"] = df

    for ds_name, sub in datasets.items():
        for m in METRIC_COLS:
            # (a) Complexity ~ Group_bin, controlling for metric M
            tmp = sub[["Complexity", "Group_bin", m]].dropna()
            if len(tmp) >= 8 and tmp[m].std() > 0:
                try:
                    res = pg.partial_corr(
                        data=tmp, x="Complexity", y="Group_bin", covar=m,
                        method="spearman",
                    )
                    r_a = float(res["r"].iloc[0])
                    p_a = float(res["p-val"].iloc[0])
                    ci_a = res["CI95%"].iloc[0]
                except Exception:
                    r_a, p_a, ci_a = np.nan, np.nan, [np.nan, np.nan]
            else:
                r_a, p_a, ci_a = np.nan, np.nan, [np.nan, np.nan]

            # (b) Complexity ~ M, controlling for Age + Sex
            tmp2 = sub[["Complexity", m, "Age", "Sex"]].dropna()
            if len(tmp2) >= 8 and tmp2[m].std() > 0:
                try:
                    res2 = pg.partial_corr(
                        data=tmp2, x="Complexity", y=m, covar=["Age", "Sex"],
                        method="spearman",
                    )
                    r_b = float(res2["r"].iloc[0])
                    p_b = float(res2["p-val"].iloc[0])
                    ci_b = res2["CI95%"].iloc[0]
                except Exception:
                    r_b, p_b, ci_b = np.nan, np.nan, [np.nan, np.nan]
            else:
                r_b, p_b, ci_b = np.nan, np.nan, [np.nan, np.nan]

            rows.append({
                "Dataset": ds_name,
                "Metric": m,
                "n_GroupCtrl": len(sub[["Complexity", "Group_bin", m]].dropna()),
                "rho_Complexity_Group_ctrlM": r_a,
                "p_Complexity_Group_ctrlM": p_a,
                "CI95_lo_ctrlM": ci_a[0] if hasattr(ci_a, "__len__") else np.nan,
                "CI95_hi_ctrlM": ci_a[1] if hasattr(ci_a, "__len__") else np.nan,
                "n_AgeSexCtrl": len(tmp2),
                "rho_Complexity_M_ctrlAgeSex": r_b,
                "p_Complexity_M_ctrlAgeSex": p_b,
                "CI95_lo_ctrlAgeSex": ci_b[0] if hasattr(ci_b, "__len__") else np.nan,
                "CI95_hi_ctrlAgeSex": ci_b[1] if hasattr(ci_b, "__len__") else np.nan,
            })

    # BH FDR correction within each analysis type across all (dataset, metric) combos
    out = pd.DataFrame(rows)
    for p_col, q_col in [
        ("p_Complexity_Group_ctrlM", "q_BH_ctrlM"),
        ("p_Complexity_M_ctrlAgeSex", "q_BH_ctrlAgeSex"),
    ]:
        pvals = out[p_col].values
        valid_mask = ~np.isnan(pvals)
        qs = np.full(len(pvals), np.nan)
        if valid_mask.sum() > 0:
            qs[valid_mask] = bh_fdr(pvals[valid_mask])
        out[q_col] = qs

    out.to_csv(RESULTS / "partial_correlations.csv", index=False)
    print(f"Partial correlations: {len(out)} rows → figures/Figure7/partial_correlations.csv")
    return out




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()

    print("\n--- Analysis 1: Spearman correlations ---")
    spear = run_spearman(df)

    print("\n--- Analysis 1b: Correlation heatmap ---")
    make_heatmap(spear)

    print("\n--- Analysis 2: Partial correlations ---")
    partial = run_partial(df)

    # Quick summary of significant results
    print("\n=== Significant Spearman correlations (q<0.05, All subjects) ===")
    sig = spear[(spear["Subset"] == "All") & (spear["q_BH"] < 0.05)].copy()
    sig = sig.sort_values(["Dataset", "rho"])
    print(sig[["Dataset", "Metric", "n", "rho", "p", "q_BH", "CI_lo", "CI_hi"]].to_string(index=False))

    print("\n=== Done ===")
