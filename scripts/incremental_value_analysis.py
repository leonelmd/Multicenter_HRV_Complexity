"""
Incremental Predictive Value of rcMSE-AUC Beyond Traditional HRV
=================================================================
Three analyses:

  1. Hierarchical logistic regression (per dataset + pooled)
       Block 1: confounders (Age, Sex where available; + Center dummies for pooled)
       Block 2: Block 1 + best traditional HRV metric (pNN50 — most consistently
                significant across all three cohorts; TINN excluded because ~53% of
                Chile and ~33% of Spain values are zero due to histogram computation
                failure on short recordings)
       Block 3: Block 2 + rcMSE-AUC (Complexity)
       Metrics: CV-AUC (5-fold stratified), Nagelkerke pseudo-R²,
                log-likelihood, ΔR², LR-test p
     → figures/Figure7/hierarchical_regression.csv

  2. Commonality analysis
       Partition Nagelkerke R² (full 4-predictor model) into:
         Unique to rcMSE-AUC | Unique to pNN50 | Shared pNN50∩Complexity |
         Unique to confounders | remaining shared terms
       Fit all 2^k − 1 submodels (k = 3 predictor groups) per dataset.
     → figures/Figure7/commonality_analysis.csv
     → figures/Figure7/Fig7_D_variance_decomposition.png

  3. Age/sex-adjusted Spearman correlations
       Regress Age + Sex out of both Complexity and each metric (OLS residuals),
       then re-compute Spearman ρ. Flag |Δρ| > 0.1.
     → figures/Figure7/age_sex_adjusted_correlations.csv

Notes on data availability:
  - Chile  : Age (62/71 available), Sex M/F (68/71 available)
  - Spain  : Age (52/58 available), Sex = entirely missing in public release
  - Japan  : Age + Sex (37/37 complete)
  - Pooled : Age + Center dummies (Sex omitted because Spain is missing)
"""

import warnings
import itertools
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "figures" / "Figure7"
FIGURES = ROOT / "figures"

BEST_TRAD = "pNN50"     # most consistent metric across all 3 cohorts (TINN excluded:
                        # ~53% zeros in Chile and ~33% in Spain due to histogram
                        # computation failure on short recordings)
CV_FOLDS  = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = p[order] * n / (np.arange(1, n + 1))
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return np.clip(q, 0, 1)


def nagelkerke_r2(y: np.ndarray, y_proba: np.ndarray) -> float:
    """Nagelkerke pseudo-R² for a logistic regression model."""
    n = len(y)
    p_bar = y.mean()
    # Null log-likelihood (intercept-only)
    ll_null = n * (p_bar * np.log(p_bar + 1e-15) + (1 - p_bar) * np.log(1 - p_bar + 1e-15))
    # Model log-likelihood
    ll_model = -n * log_loss(y, y_proba)
    # Cox-Snell
    cs = 1 - np.exp(2 / n * (ll_null - ll_model))
    cs_max = 1 - np.exp(2 / n * ll_null)
    return float(cs / cs_max) if cs_max > 0 else np.nan


def log_likelihood(y: np.ndarray, y_proba: np.ndarray) -> float:
    n = len(y)
    return float(-n * log_loss(y, y_proba))


def null_log_likelihood(y: np.ndarray) -> float:
    n = len(y)
    p_bar = y.mean()
    return float(n * (p_bar * np.log(p_bar + 1e-15) + (1 - p_bar) * np.log(1 - p_bar + 1e-15)))


def lr_test(ll_restricted: float, ll_full: float, df_diff: int):
    """Likelihood ratio test. Returns (chi2_stat, p_value)."""
    chi2 = 2 * (ll_full - ll_restricted)
    p = stats.chi2.sf(chi2, df=df_diff)
    return float(chi2), float(p)


def fit_logistic(X: np.ndarray, y: np.ndarray, cv: int = CV_FOLDS):
    """
    Fit logistic regression with StandardScaler.
    Returns: model (fit on full data), cv_auc, y_proba (full data).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")),
    ])
    cv_aucs = cross_val_score(
        pipe, X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
    )
    pipe.fit(X, y)
    y_proba = pipe.predict_proba(X)[:, 1]
    return pipe, float(cv_aucs.mean()), float(cv_aucs.std()), y_proba


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    hrv  = pd.read_csv(RESULTS / "traditional_hrv_metrics.csv")
    cons = pd.read_csv(RESULTS / "consolidated_metrics_no_imputation.csv")[
        ["Subject", "Center", "Complexity"]
    ]

    # Demographics
    chile_demo = (pd.read_csv(DATA / "chile_demographics.csv")
                  .rename(columns={"Anon_ID": "Subject"})[["Subject", "Age", "Gender"]])
    chile_demo["Sex"] = chile_demo["Gender"].map({"M": 0, "F": 1})

    spain_demo = pd.read_csv(DATA / "spain_demographics.csv")[["Subject", "Age", "Gender"]]
    spain_demo["Sex"] = np.nan   # entirely missing in public release

    japan_meta = (pd.read_csv(DATA / "japan_metadata.csv")
                  .rename(columns={"Subject_ID": "Subject"})[["Subject", "Age", "Gender"]])
    japan_meta["Gender"] = japan_meta["Gender"].map({0: "Male", 1: "Female"})
    japan_meta["Sex"] = japan_meta["Gender"].str.lower().map({"male": 0, "female": 1})

    demo = pd.concat(
        [chile_demo[["Subject", "Age", "Sex"]],
         spain_demo[["Subject", "Age", "Sex"]],
         japan_meta[["Subject", "Age", "Sex"]]],
        ignore_index=True,
    )

    df = hrv.merge(cons[["Subject", "Complexity"]], on="Subject", how="inner")
    df = df.merge(demo, on="Subject", how="left")
    df["Group_bin"] = df["Group"].map({"PD": 1, "Control": 0})

    return df


# ============================================================
# ANALYSIS 1 — Hierarchical logistic regression
# ============================================================

def hierarchical_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit three nested logistic regression blocks per dataset.
    Block 1: confounders only
    Block 2: confounders + pNN50
    Block 3: confounders + pNN50 + Complexity
    """
    rows = []

    datasets = {
        "Chile":  df[df["Center"] == "Chile"],
        "Spain":  df[df["Center"] == "Spain"],
        "Japan":  df[df["Center"] == "Japan"],
        "Pooled": df,
    }

    for ds_name, sub in datasets.items():
        y = sub["Group_bin"]

        # Confounders
        if ds_name == "Pooled":
            # Use Age + Center dummies (Sex omitted: Spain entirely missing)
            sub = sub.copy()
            sub["Center_Spain"] = (sub["Center"] == "Spain").astype(float)
            sub["Center_Japan"] = (sub["Center"] == "Japan").astype(float)
            conf_cols = ["Age", "Center_Spain", "Center_Japan"]
        elif ds_name == "Spain":
            conf_cols = ["Age"]       # Sex not available
        else:
            conf_cols = ["Age", "Sex"]

        # Drop rows missing any required variable across all three blocks
        needed = conf_cols + [BEST_TRAD, "Complexity", "Group_bin"]
        mask = sub[needed].notna().all(axis=1)
        sub_c = sub[mask].copy()
        y_c   = sub_c["Group_bin"].values

        if len(sub_c) < 10 or y_c.sum() < 3 or (len(y_c) - y_c.sum()) < 3:
            print(f"  Skipping {ds_name}: insufficient complete cases ({len(sub_c)})")
            continue

        print(f"  {ds_name}: n={len(sub_c)} ({int(y_c.sum())} PD / {int(len(y_c)-y_c.sum())} HC)")

        ll_null = null_log_likelihood(y_c)

        blocks = {
            "Block1_Confounders": conf_cols,
            "Block2_Conf+pNN50":   conf_cols + [BEST_TRAD],
            "Block3_Conf+pNN50+Complexity": conf_cols + [BEST_TRAD, "Complexity"],
        }

        prev_ll   = ll_null
        prev_r2   = 0.0
        prev_npar = 0

        for block_name, feat_cols in blocks.items():
            X = sub_c[feat_cols].values.astype(float)
            n_added = len(feat_cols) - prev_npar

            _, cv_auc, cv_auc_sd, y_proba = fit_logistic(X, y_c)
            ll = log_likelihood(y_c, y_proba)
            r2 = nagelkerke_r2(y_c, y_proba)
            delta_r2 = r2 - prev_r2
            chi2, p_lr = lr_test(prev_ll, ll, df_diff=n_added)

            rows.append({
                "Dataset":   ds_name,
                "Block":     block_name,
                "n":         len(sub_c),
                "n_PD":      int(y_c.sum()),
                "predictors": ", ".join(feat_cols),
                "n_params":  len(feat_cols),
                "CV_AUC":    cv_auc,
                "CV_AUC_SD": cv_auc_sd,
                "Nagelkerke_R2": r2,
                "Delta_R2":  delta_r2,
                "LogLik":    ll,
                "LR_chi2":   chi2,
                "LR_p":      p_lr,
            })
            prev_ll   = ll
            prev_r2   = r2
            prev_npar = len(feat_cols)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "hierarchical_regression.csv", index=False)
    print(f"\nHierarchical regression → figures/Figure7/hierarchical_regression.csv")
    return out


def print_hier_table(hier: pd.DataFrame):
    print("\n=== Hierarchical Logistic Regression Summary ===")
    print(f"{'Dataset':<8} {'Block':<36} {'n':>4} {'CV-AUC':>8} {'NagelR2':>9} "
          f"{'ΔR2':>7} {'LR_p':>10}")
    print("-" * 90)
    for _, r in hier.iterrows():
        p_str = f"{r.LR_p:.4f}" if not np.isnan(r.LR_p) else "—"
        print(f"{r.Dataset:<8} {r.Block:<36} {r.n:>4} "
              f"{r.CV_AUC:>8.3f} {r.Nagelkerke_R2:>9.4f} "
              f"{r.Delta_R2:>7.4f} {p_str:>10}")


# ============================================================
# ANALYSIS 2 — Commonality analysis
# ============================================================

def mcfadden_r2(y: np.ndarray, feat_cols: list[str], sub: pd.DataFrame) -> float:
    """McFadden pseudo-R² for a given set of features on sub."""
    X = sub[feat_cols].values.astype(float)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    clf.fit(X_s, y)
    y_proba = clf.predict_proba(X_s)[:, 1]
    ll_model = log_likelihood(y, y_proba)
    ll_null  = null_log_likelihood(y)
    return float(1 - ll_model / ll_null)


def commonality_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three predictor groups:
      A = confounders (Age + Sex or Age)
      B = pNN50
      C = Complexity
    Fit 7 submodels (all non-empty subsets of {A,B,C}).
    Decompose Nagelkerke R² into 7 commonality components.
    """
    rows = []
    datasets = {
        "Chile":  df[df["Center"] == "Chile"],
        "Spain":  df[df["Center"] == "Spain"],
        "Japan":  df[df["Center"] == "Japan"],
        "Pooled": df,
    }

    for ds_name, sub in datasets.items():
        if ds_name == "Pooled":
            sub = sub.copy()
            sub["Center_Spain"] = (sub["Center"] == "Spain").astype(float)
            sub["Center_Japan"] = (sub["Center"] == "Japan").astype(float)
            A_cols = ["Age", "Center_Spain", "Center_Japan"]
        elif ds_name == "Spain":
            A_cols = ["Age"]
        else:
            A_cols = ["Age", "Sex"]

        B_cols = [BEST_TRAD]
        C_cols = ["Complexity"]

        needed = A_cols + B_cols + C_cols + ["Group_bin"]
        mask = sub[needed].notna().all(axis=1)
        sub_c = sub[mask].copy()
        y = sub_c["Group_bin"].values

        if len(sub_c) < 10:
            continue

        # All 7 non-empty subsets of {A, B, C}
        group_map = {"A": A_cols, "B": B_cols, "C": C_cols}
        r2 = {}
        for k in range(1, 4):
            for combo in itertools.combinations(["A", "B", "C"], k):
                cols = []
                for g in combo:
                    cols += group_map[g]
                key = "".join(combo)
                r2[key] = mcfadden_r2(y, cols, sub_c)

        # Commonality decomposition (inclusion-exclusion)
        # Unique contributions
        U_A = r2["ABC"] - r2["BC"]
        U_B = r2["ABC"] - r2["AC"]
        U_C = r2["ABC"] - r2["AB"]
        # Pairwise exclusive shared
        S_AB = r2["BC"] + r2["AC"] - r2["ABC"] - r2["C"]
        S_AC = r2["BC"] + r2["AB"] - r2["ABC"] - r2["B"]
        S_BC = r2["AC"] + r2["AB"] - r2["ABC"] - r2["A"]
        # Triple shared
        S_ABC = r2["ABC"] - r2["BC"] - r2["AC"] - r2["AB"] + r2["A"] + r2["B"] + r2["C"]
        total = U_A + U_B + U_C + S_AB + S_AC + S_BC + S_ABC

        rows.append({
            "Dataset":          ds_name,
            "n":                len(sub_c),
            "R2_full":          r2["ABC"],
            "R2_A":             r2["A"],
            "R2_B":             r2["B"],
            "R2_C":             r2["C"],
            "R2_AB":            r2["AB"],
            "R2_AC":            r2["AC"],
            "R2_BC":            r2["BC"],
            "Unique_Confounders": U_A,
            "Unique_pNN50":      U_B,
            "Unique_Complexity": U_C,
            "Shared_Conf_pNN50": S_AB,
            "Shared_Conf_Compl": S_AC,
            "Shared_pNN50_Compl": S_BC,
            "Shared_Triple":    S_ABC,
            "Sum_check":        total,
        })
        print(f"  {ds_name}: R²(full)={r2['ABC']:.4f}  "
              f"Unique_Complexity={U_C:.4f}  Unique_pNN50={U_B:.4f}  "
              f"Shared_pNN50×Compl={S_BC:.4f}  Unique_Conf={U_A:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "commonality_analysis.csv", index=False)
    print(f"Commonality analysis → figures/Figure7/commonality_analysis.csv")
    return out


def make_variance_decomp_plot(comm: pd.DataFrame):
    """Stacked bar chart of variance partition per dataset."""
    ds_order = [d for d in ["Chile", "Spain", "Japan", "Pooled"] if d in comm["Dataset"].values]

    components = [
        ("Unique_Complexity",  "#1565C0", "Unique to rcMSE-AUC"),
        ("Unique_pNN50",        "#2E7D32", "Unique to pNN50"),
        ("Shared_pNN50_Compl",  "#7B1FA2", "Shared pNN50 ∩ rcMSE-AUC"),
        ("Unique_Confounders", "#E65100", "Unique to Confounders"),
        ("Shared_Conf_pNN50",   "#F9A825", "Shared Conf ∩ pNN50"),
        ("Shared_Conf_Compl",  "#00838F", "Shared Conf ∩ rcMSE-AUC"),
        ("Shared_Triple",      "#616161", "Triple shared"),
    ]

    fig, axes = plt.subplots(1, len(ds_order), figsize=(12, 5), sharey=False)
    if len(ds_order) == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, ds_order):
        row = comm[comm["Dataset"] == ds_name].iloc[0]
        vals  = [row[c] for c, _, _ in components]
        cols  = [col  for _, col, _ in components]
        lbls  = [lbl  for _, _, lbl in components]

        # Clip negative values (numerical noise) but keep them visually
        vals_clip = [max(v, 0) for v in vals]
        total = sum(vals_clip)

        # Normalise to fraction of full R²
        r2_full = row["R2_full"]
        fracs = [v / r2_full if r2_full > 0 else 0 for v in vals_clip]

        bottom = 0.0
        for frac, color, lbl in zip(fracs, cols, lbls):
            if frac > 1e-4:
                ax.bar(0, frac, bottom=bottom, color=color, label=lbl,
                       width=0.6, edgecolor="white", linewidth=0.5)
                if frac > 0.04:
                    ax.text(0, bottom + frac / 2, f"{frac*100:.1f}%",
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
                bottom += frac

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1.15)
        ax.set_xticks([])
        ax.set_title(f"{ds_name}\nn={int(row['n'])}", fontsize=10)
        ax.set_ylabel("Fraction of McFadden R² (full model)" if ax is axes[0] else "")
        ax.text(0, 1.05, f"R²={r2_full:.3f}", ha="center", va="bottom",
                fontsize=8, color="black")

    # Legend outside
    handles = [mpatches.Patch(color=c, label=l) for _, c, l in components]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.18))
    fig.suptitle(
        "McFadden R² Commonality Decomposition\n"
        f"Full model: Confounders + {BEST_TRAD} + rcMSE-AUC",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    out = FIGURES / "Figure7" / "Fig7_D_variance_decomposition.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Variance decomposition → {out}")


# ============================================================
# ANALYSIS 3 — Age/sex-adjusted Spearman correlations
# ============================================================

def adjusted_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """
    Partial out Age + Sex (OLS residuals) before computing Spearman ρ.
    Compare adjusted ρ to unadjusted values from spearman_correlations.csv.
    """
    from sklearn.linear_model import LinearRegression

    METRIC_COLS = [
        "SDNN", "RMSSD", "pNN50", "SDANN",
        # TINN excluded: spurious zeros on short recordings
        "SD1", "SD2", "SD1SD2",
        "VLF_power", "LF_power", "HF_power", "Total_power",
        "LF_HF", "LF_norm", "HF_norm",
        "DFA_alpha1", "DFA_alpha2",
        # SampEn excluded: it is RC-MSE Scale=1, a direct component of rcMSE-AUC
    ]

    unadj = pd.read_csv(RESULTS / "spearman_correlations.csv")
    unadj = unadj[(unadj["Subset"] == "All")][["Dataset", "Metric", "rho", "p", "q_BH"]]
    unadj = unadj.rename(columns={"rho": "rho_unadj", "p": "p_unadj", "q_BH": "q_unadj"})

    datasets = {
        "Chile":  df[df["Center"] == "Chile"],
        "Spain":  df[df["Center"] == "Spain"],
        "Japan":  df[df["Center"] == "Japan"],
        "Pooled": df,
    }

    rows = []
    for ds_name, sub in datasets.items():
        if ds_name == "Pooled":
            sub = sub.copy()
            sub["Center_Spain"] = (sub["Center"] == "Spain").astype(float)
            sub["Center_Japan"] = (sub["Center"] == "Japan").astype(float)
            covar_cols = ["Age", "Center_Spain", "Center_Japan"]
        elif ds_name == "Spain":
            covar_cols = ["Age"]
        else:
            covar_cols = ["Age", "Sex"]

        for metric in METRIC_COLS:
            needed = covar_cols + [metric, "Complexity"]
            sub_c = sub[needed].dropna()
            if len(sub_c) < 8:
                rows.append({"Dataset": ds_name, "Metric": metric,
                             "n_adjusted": len(sub_c),
                             "rho_adj": np.nan, "p_adj": np.nan, "q_adj": np.nan,
                             "delta_rho": np.nan, "flag": ""})
                continue

            C = sub_c[covar_cols].values.astype(float)
            C_scaled = StandardScaler().fit_transform(C)

            # Residualise metric
            lr_m = LinearRegression().fit(C_scaled, sub_c[metric].values)
            resid_m = sub_c[metric].values - lr_m.predict(C_scaled)

            # Residualise Complexity
            lr_c = LinearRegression().fit(C_scaled, sub_c["Complexity"].values)
            resid_c = sub_c["Complexity"].values - lr_c.predict(C_scaled)

            rho_adj, p_adj = stats.spearmanr(resid_m, resid_c)
            rows.append({"Dataset": ds_name, "Metric": metric,
                         "n_adjusted": len(sub_c),
                         "rho_adj": float(rho_adj), "p_adj": float(p_adj),
                         "q_adj": np.nan,   # fill after BH below
                         "delta_rho": np.nan, "flag": ""})

    out = pd.DataFrame(rows)

    # BH across all (Dataset, Metric) for adjusted p
    pvals = out["p_adj"].values
    valid = ~np.isnan(pvals)
    qs = np.full(len(pvals), np.nan)
    if valid.sum() > 0:
        qs[valid] = bh_fdr(pvals[valid])
    out["q_adj"] = qs

    # Merge unadjusted ρ and compute Δρ
    out = out.merge(unadj, on=["Dataset", "Metric"], how="left")
    out["delta_rho"] = out["rho_adj"] - out["rho_unadj"]
    out["flag"] = out["delta_rho"].abs().gt(0.1).map({True: "FLAG", False: ""})

    out.to_csv(RESULTS / "age_sex_adjusted_correlations.csv", index=False)
    print(f"Age/sex-adjusted correlations → figures/Figure7/age_sex_adjusted_correlations.csv")

    # Print flagged
    flagged = out[out["flag"] == "FLAG"].sort_values("delta_rho")
    if len(flagged):
        print("\n=== Correlations with |Δρ| > 0.1 after age/sex adjustment ===")
        print(flagged[["Dataset", "Metric", "rho_unadj", "rho_adj", "delta_rho"]].to_string(index=False))
    else:
        print("  No correlations changed by |Δρ| > 0.1 after age/sex adjustment.")

    return out


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Dataset: {len(df)} subjects across {df['Center'].nunique()} centers\n")

    print("=== Analysis 1: Hierarchical logistic regression ===")
    hier = hierarchical_regression(df)
    print_hier_table(hier)

    print("\n=== Analysis 2: Commonality analysis ===")
    comm = commonality_analysis(df)
    make_variance_decomp_plot(comm)

    print("\n=== Analysis 3: Age/sex-adjusted Spearman correlations ===")
    adj = adjusted_spearman(df)

    # Print final block-3 summary (incremental AUC from adding Complexity)
    print("\n=== Incremental AUC gain: Block 2 → Block 3 (adding rcMSE-AUC) ===")
    b2 = hier[hier["Block"].str.startswith("Block2")].set_index("Dataset")["CV_AUC"]
    b3 = hier[hier["Block"].str.startswith("Block3")].set_index("Dataset")["CV_AUC"]
    for ds in b2.index:
        delta = b3[ds] - b2[ds]
        print(f"  {ds:<8}: AUC {b2[ds]:.3f} → {b3[ds]:.3f}  (Δ={delta:+.3f})")

    print("\n=== Done ===")
