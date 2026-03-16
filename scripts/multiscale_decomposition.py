"""
Per-Scale Entropy vs Traditional HRV Metrics — Multiscale Decomposition
========================================================================
Maps which time scales of complexity loss correspond to which autonomic
mechanisms, using pre-computed per-scale SampEn from the RC-MSE toolbox.

Per-scale entropy data (already computed by Julia RC-MSE toolbox):
  - data/chile_mse.csv            — 71 subjects × 20 scales (~15 min rest ECG)
  - data/spain_mse.csv            — 58 subjects × 20 scales (5–15 min PPG)
  - data/japan_morning_mse.csv    — 39 subjects × 20 scales (07–11h window)
  - data/japan_afternoon_mse.csv  — 45 subjects × 20 scales (16–20h window, best)

For short recordings (CETRAM/Cruces): scales 1–5 are physiologically
interpretable; scales 6–20 are computed but must be interpreted cautiously.
For Nagoya 24h recordings: all 20 scales are meaningful.

Analyses:
  1. Scale-metric correlation heatmap → figures/Figure8/Fig8_A_correlation_heatmap.png (via complexity_correlation_analysis.py)
                                      → figures/Figure8/scale_metric_correlations.csv
  2. Annotated MSE curves (PD vs HC with Mann-Whitney per scale)
                                      → figures/Figure8/Fig8_E_mse_curves_annotated.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "figures" / "Figure8"
FIGURES = ROOT / "figures"

# Physiological zone boundaries (scale numbers, inclusive)
VAGAL_ZONE  = (1, 5)    # beat-to-beat, vagal; tracks RMSSD, HF, SD1
BARO_ZONE   = (6, 15)   # respiratory / baroreflex; tracks LF, LF/HF
SLOW_ZONE   = (16, 20)  # slow autonomic / circadian (24 h only); tracks SDANN, VLF, DFA_α2

ZONE_STYLE = {
    "Vagal\n(1–5)":     {"color": "#BBDEFB", "scales": VAGAL_ZONE},
    "Baroreflex\n(6–15)": {"color": "#C8E6C9", "scales": BARO_ZONE},
    "Slow/Circ.\n(16–20)": {"color": "#FFE0B2", "scales": SLOW_ZONE},
}

# Metrics to include in the heatmap
# Excluded: SampEn (it IS scale-1 entropy — circular)
# Excluded: TINN (~53% zeros in Chile, ~33% in Spain due to histogram computation failure
#           on short recordings; produces spurious correlations)
HEATMAP_METRICS = [
    "RMSSD", "SDNN", "pNN50",
    "SD1", "SD2", "SD1SD2",
    "VLF_power", "LF_power", "HF_power", "Total_power",
    "LF_HF", "LF_norm", "HF_norm",
    "DFA_alpha1", "DFA_alpha2", "SDANN",
]

# Friendly y-axis labels
METRIC_LABELS = {
    "RMSSD": "RMSSD", "SDNN": "SDNN", "pNN50": "pNN50",
    "SD1": "SD1", "SD2": "SD2", "SD1SD2": "SD1/SD2",
    "VLF_power": "VLF power", "LF_power": "LF power", "HF_power": "HF power",
    "Total_power": "Total power",
    "LF_HF": "LF/HF ratio", "LF_norm": "LFnu", "HF_norm": "HFnu",
    "DFA_alpha1": "DFA α1", "DFA_alpha2": "DFA α2", "SDANN": "SDANN",
}

PD_COLOR  = "#C62828"
HC_COLOR  = "#1565C0"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def stars(q: float) -> str:
    if np.isnan(q):   return ""
    if q < 0.001:     return "***"
    if q < 0.01:      return "**"
    if q < 0.05:      return "*"
    return ""


def load_mse(path: Path, label: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Normalise column names
    df = df.rename(columns={"MSE": "Entropy", "Scales": "Scale"})
    df["Group"] = df["Group"].str.strip().str.lower().map(
        lambda g: "PD" if g == "pd" else "Control"
    )
    if label:
        df["Window"] = label
    return df[["Subject", "Group", "Scale", "Entropy"] + (["Window"] if label else [])]


def add_zone_bands(ax, xlim=(0.5, 20.5), alpha=0.18, short_recording=False):
    """Shade physiological zone regions on an axis with scale on x-axis."""
    for name, style in ZONE_STYLE.items():
        lo, hi = style["scales"]
        ax.axvspan(lo - 0.5, hi + 0.5, alpha=alpha, color=style["color"], zorder=0)
        # Fade slow zone for short recordings
        if short_recording and lo >= SLOW_ZONE[0]:
            ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.3, color="grey", zorder=1,
                       hatch="//", fill=False)


def add_zone_bands_on_heatmap(ax, n_metrics: int, alpha=0.25, short_recording=False):
    """Add vertical zone bands to a heatmap (y = metric index, x = scale index)."""
    for name, style in ZONE_STYLE.items():
        lo, hi = style["scales"]
        ax.axvline(lo - 1, color="grey", lw=0.5, ls=":")
        ax.axvline(hi,     color="grey", lw=0.5, ls=":")
        # background span (x = scale index 0-based)
        ax.axvspan(lo - 1 - 0.5, hi - 0.5, alpha=alpha, color=style["color"], zorder=0)
        if short_recording and lo >= SLOW_ZONE[0]:
            ax.axvspan(lo - 1 - 0.5, hi - 0.5, alpha=0.3, color="grey",
                       zorder=1, hatch="//", fill=False)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data():
    hrv = pd.read_csv(RESULTS / "traditional_hrv_metrics.csv")

    chile_mse  = load_mse(DATA / "chile_mse.csv")
    chile_mse["Center"] = "Chile"

    spain_mse  = load_mse(DATA / "spain_mse.csv")
    spain_mse["Center"] = "Spain"

    japan_morning_mse   = load_mse(DATA / "japan_morning_mse.csv")
    japan_morning_mse["Center"] = "Japan"

    japan_afternoon_mse = load_mse(DATA / "japan_afternoon_mse.csv")
    japan_afternoon_mse["Center"] = "Japan"

    # Merge each MSE dataset with the corresponding HRV metrics
    def merge_with_hrv(mse_df):
        return mse_df.merge(
            hrv[["Subject", "Center"] + HEATMAP_METRICS],
            on=["Subject", "Center"], how="inner",
        )

    data = {
        "Chile":            merge_with_hrv(chile_mse),
        "Spain":            merge_with_hrv(spain_mse),
        "Japan-morning":    merge_with_hrv(japan_morning_mse),
        "Japan-afternoon":  merge_with_hrv(japan_afternoon_mse),
    }

    # Mean IBI per dataset for time-axis annotation (ms)
    mean_ibi = (
        hrv.groupby("Center")["MeanNN"].mean()
        .rename({"Chile": "Chile", "Spain": "Spain", "Japan": "Japan"})
    )

    return data, hrv, mean_ibi


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1 — Scale-metric correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def scale_metric_correlations(data: dict) -> pd.DataFrame:
    """Compute Spearman ρ(entropy_at_scale_s, metric_M) per dataset."""
    rows = []
    for ds_name, df in data.items():
        for scale in range(1, 21):
            sub = df[df["Scale"] == scale]
            ent = sub["Entropy"].values
            # Collect raw p-values for this (dataset, scale) slice
            raw_p = {}
            for m in HEATMAP_METRICS:
                vals = sub[m].values
                mask = np.isfinite(ent) & np.isfinite(vals)
                if mask.sum() < 5:
                    raw_p[m] = np.nan
                    rows.append({"Dataset": ds_name, "Scale": scale, "Metric": m,
                                 "n": int(mask.sum()), "rho": np.nan,
                                 "p": np.nan, "q_BH": np.nan, "sig": ""})
                else:
                    rho, p = stats.spearmanr(ent[mask], vals[mask])
                    raw_p[m] = p
                    rows.append({"Dataset": ds_name, "Scale": scale, "Metric": m,
                                 "n": int(mask.sum()), "rho": float(rho),
                                 "p": float(p), "q_BH": np.nan, "sig": ""})

        # BH-FDR across ALL (scale, metric) pairs within this dataset
        idxs = [i for i, r in enumerate(rows) if r["Dataset"] == ds_name]
        ps   = np.array([rows[i]["p"] for i in idxs])
        valid = ~np.isnan(ps)
        qs = np.full(len(ps), np.nan)
        if valid.sum() > 0:
            qs[valid] = bh_fdr(ps[valid])
        for i, idx in enumerate(idxs):
            rows[idx]["q_BH"] = qs[i]
            rows[idx]["sig"]  = stars(qs[i])

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "scale_metric_correlations.csv", index=False)
    print(f"Scale-metric correlations: {len(out)} rows → figures/Figure8/scale_metric_correlations.csv")
    return out


def make_heatmap(corr_df: pd.DataFrame, mean_ibi: pd.Series):
    """
    One heatmap panel per dataset. Rows = metrics, columns = scales.
    Secondary x-axis = approximate time resolution (s). Zone bands.
    """
    datasets = [
        ("Chile",           mean_ibi.get("Chile", 837), True,  "CETRAM  ·  ECG ~15 min"),
        ("Spain",           mean_ibi.get("Spain", 894), True,  "Cruces  ·  PPG 5–15 min"),
        ("Japan-morning",   mean_ibi.get("Japan", 803), False, "Nagoya 07–11h  ·  ECG 4 h"),
        ("Japan-afternoon", mean_ibi.get("Japan", 803), False, "Nagoya 16–20h  ·  ECG 4 h"),
    ]
    n_panels = len(datasets)
    fig, axes = plt.subplots(1, n_panels, figsize=(22, 8),
                             gridspec_kw={"wspace": 0.4})

    # Shared colour scale
    vmax = 0.7

    for ax, (ds_name, ibi_ms, short, subtitle) in zip(axes, datasets):
        sub = corr_df[corr_df["Dataset"] == ds_name]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Build rho / annotation matrices
        rho_mat = np.full((len(HEATMAP_METRICS), 20), np.nan)
        ann_mat = [[""] * 20 for _ in HEATMAP_METRICS]
        for _, row in sub.iterrows():
            r = HEATMAP_METRICS.index(row["Metric"]) if row["Metric"] in HEATMAP_METRICS else None
            c = int(row["Scale"]) - 1
            if r is None:
                continue
            if not np.isnan(row["rho"]):
                rho_mat[r, c] = row["rho"]
                ann_mat[r][c] = row["sig"]

        ylabels = [METRIC_LABELS.get(m, m) for m in HEATMAP_METRICS]
        xlabels = [str(s) for s in range(1, 21)]

        sns.heatmap(
            rho_mat, ax=ax,
            cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
            linewidths=0.3, linecolor="#e0e0e0",
            annot=np.array(ann_mat), fmt="s",
            annot_kws={"size": 7, "weight": "bold"},
            cbar_kws={"label": "Spearman ρ", "shrink": 0.7},
            xticklabels=xlabels, yticklabels=ylabels,
        )

        # Zone bands (axvspan on heatmap — columns 0-based)
        for _, style in ZONE_STYLE.items():
            lo, hi = style["scales"]
            ax.axvspan(lo - 1, hi, alpha=0.12, color=style["color"], zorder=0)
            if short and lo >= SLOW_ZONE[0]:
                ax.axvspan(lo - 1, hi, alpha=0.28, color="#bdbdbd", zorder=1,
                           hatch="//", fill=False)

        # Secondary x-axis: time resolution in seconds
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_scales = [1, 5, 10, 15, 20]
        ax2.set_xticks([s - 0.5 for s in tick_scales])
        ax2.set_xticklabels([f"{s * ibi_ms / 1000:.1f}s" for s in tick_scales], fontsize=7)
        ax2.set_xlabel("Approx. time resolution", fontsize=8, labelpad=4)

        ax.set_title(f"{ds_name}\n{subtitle}", fontsize=9, fontweight="bold", pad=24)
        ax.set_xlabel("Scale", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7.5)

    # Zone legend
    handles = [
        mpatches.Patch(color=s["color"], alpha=0.6, label=name.replace("\n", " "))
        for name, s in ZONE_STYLE.items()
    ] + [mpatches.Patch(color="#bdbdbd", alpha=0.5, hatch="//",
                        label="Unreliable (short recording)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Spearman ρ: Entropy at Each MSE Scale vs Traditional HRV Metrics\n"
        "(* q<0.05  ** q<0.01  *** q<0.001, BH-FDR across all scale×metric pairs)",
        fontsize=11, y=1.01,
    )

    out = FIGURES / "Figure8" / "scale_physiology_heatmap_4panel.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scale heatmap (4-panel reference) → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2 — Scale-specific group differences (MSE curves)
# ─────────────────────────────────────────────────────────────────────────────

def make_mse_curves(data: dict):
    """
    Mean ± SE entropy curves for PD vs HC per dataset, with Mann-Whitney
    significance markers (BH-FDR corrected) and physiological zone bands.
    """
    panel_order = ["Chile", "Spain", "Japan-morning", "Japan-afternoon"]
    short_flag  = {"Chile": True, "Spain": True, "Japan-morning": False, "Japan-afternoon": False}
    titles = {
        "Chile":            "CETRAM (ECG · ~15 min)",
        "Spain":            "Cruces (PPG · 5–15 min)",
        "Japan-morning":    "Nagoya 07–11h (ECG · 4 h)",
        "Japan-afternoon":  "Nagoya 16–20h (ECG · 4 h)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes_flat = axes.flatten()

    scale_group_stats = []  # for summary output

    for ax, ds_name in zip(axes_flat, panel_order):
        if ds_name not in data:
            ax.set_visible(False)
            continue

        df = data[ds_name]
        scales = sorted(df["Scale"].unique())
        short = short_flag[ds_name]

        # Zone bands (behind everything)
        for name, style in ZONE_STYLE.items():
            lo, hi = style["scales"]
            ax.axvspan(lo, hi, alpha=0.12, color=style["color"], zorder=0)
            if short and lo >= SLOW_ZONE[0]:
                ax.axvspan(lo, hi, alpha=0.28, color="#bdbdbd", zorder=1,
                           hatch="//", fill=False)

        # Mark "unreliable" zone boundary for short recordings
        if short:
            ax.axvline(VAGAL_ZONE[1] + 0.5, color="grey", lw=1.2, ls="--", alpha=0.7,
                       zorder=2, label="Short-recording limit")

        hc_means, hc_sems = [], []
        pd_means, pd_sems = [], []
        mw_ps = []

        for s in scales:
            sc = df[df["Scale"] == s]
            hc = sc[sc["Group"] == "Control"]["Entropy"].dropna()
            pd_ = sc[sc["Group"] == "PD"]["Entropy"].dropna()

            hc_means.append(hc.mean()); hc_sems.append(hc.sem())
            pd_means.append(pd_.mean()); pd_sems.append(pd_.sem())

            if len(hc) >= 3 and len(pd_) >= 3:
                _, p = stats.mannwhitneyu(hc, pd_, alternative="two-sided")
            else:
                p = np.nan
            mw_ps.append(p)

            scale_group_stats.append({
                "Dataset": ds_name, "Scale": s,
                "HC_mean": hc.mean(), "HC_sem": hc.sem(), "HC_n": len(hc),
                "PD_mean": pd_.mean(), "PD_sem": pd_.sem(), "PD_n": len(pd_),
                "MW_p": p, "q_BH": np.nan,
            })

        # BH-FDR within dataset
        ps_arr = np.array(mw_ps)
        valid  = ~np.isnan(ps_arr)
        qs_arr = np.full(len(ps_arr), np.nan)
        if valid.sum() > 0:
            qs_arr[valid] = bh_fdr(ps_arr[valid])

        # Update q in stats list
        for i, entry in enumerate(
            [r for r in scale_group_stats if r["Dataset"] == ds_name][-len(scales):]
        ):
            entry["q_BH"] = qs_arr[i]

        scales_arr   = np.array(scales)
        hc_means_arr = np.array(hc_means)
        hc_sems_arr  = np.array(hc_sems)
        pd_means_arr = np.array(pd_means)
        pd_sems_arr  = np.array(pd_sems)

        # Curves
        ax.plot(scales_arr, hc_means_arr, color=HC_COLOR, lw=2, label="HC")
        ax.fill_between(scales_arr,
                        hc_means_arr - hc_sems_arr,
                        hc_means_arr + hc_sems_arr,
                        color=HC_COLOR, alpha=0.2)

        ax.plot(scales_arr, pd_means_arr, color=PD_COLOR, lw=2, ls="--", label="PD")
        ax.fill_between(scales_arr,
                        pd_means_arr - pd_sems_arr,
                        pd_means_arr + pd_sems_arr,
                        color=PD_COLOR, alpha=0.2)

        # Significance markers above the max curve
        y_top = max(
            np.nanmax(hc_means_arr + hc_sems_arr),
            np.nanmax(pd_means_arr + pd_sems_arr),
        )
        y_margin = (y_top - np.nanmin(np.concatenate([hc_means_arr, pd_means_arr]))) * 0.08
        for i, (s, q) in enumerate(zip(scales_arr, qs_arr)):
            sig = stars(q)
            if sig:
                ax.text(s, y_top + y_margin, sig,
                        ha="center", va="bottom", fontsize=7.5, color="black",
                        fontweight="bold")

        ax.set_xlim(0.5, 20.5)
        ax.set_xticks(scales_arr)
        ax.set_xticklabels([str(s) if s % 5 == 0 or s == 1 else "" for s in scales_arr],
                           fontsize=8)
        ax.set_xlabel("Scale", fontsize=9)
        ax.set_ylabel("Sample Entropy", fontsize=9)
        ax.set_title(titles[ds_name], fontsize=9, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

    # Zone legend
    zone_handles = [
        mpatches.Patch(color=s["color"], alpha=0.6, label=name.replace("\n", " "))
        for name, s in ZONE_STYLE.items()
    ]
    line_handles = [
        plt.Line2D([0], [0], color=HC_COLOR, lw=2, label="HC (mean ± SE)"),
        plt.Line2D([0], [0], color=PD_COLOR, lw=2, ls="--", label="PD (mean ± SE)"),
    ]
    fig.legend(handles=zone_handles + line_handles,
               loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "MSE Curves: PD vs HC per Dataset\n"
        "(significance markers: BH-FDR corrected Mann-Whitney U)",
        fontsize=11,
    )
    plt.tight_layout()

    out = FIGURES / "Figure8" / "Fig8_E_mse_curves_annotated.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"MSE curves → {out}")

    # Save per-scale group stats
    stats_df = pd.DataFrame(scale_group_stats)
    stats_df.to_csv(RESULTS / "scale_group_differences.csv", index=False)
    print(f"Scale group differences → figures/Figure8/scale_group_differences.csv")
    return stats_df



# NOTE: Circadian decomposition of the Nagoya cohort was moved to
# scripts/generate_figure3.py, which reads data/japan_evolution.csv directly.

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    data, hrv, mean_ibi = load_all_data()
    for ds, df in data.items():
        n_subj = df["Subject"].nunique()
        print(f"  {ds}: {n_subj} subjects × {df['Scale'].nunique()} scales")

    print("\n=== Analysis 1: Scale-metric correlation heatmap ===")
    corr_df = scale_metric_correlations(data)
    make_heatmap(corr_df, mean_ibi)

    # Print top 5 (scale, metric) correlations per dataset
    print("\n=== Top scale-metric correlations (|ρ| > 0.4, q<0.05) ===")
    top = (corr_df[corr_df["q_BH"] < 0.05]
           .assign(abs_rho=corr_df["rho"].abs())
           .query("abs_rho > 0.4")
           .sort_values("abs_rho", ascending=False))
    if len(top):
        print(top[["Dataset","Scale","Metric","n","rho","q_BH"]].to_string(index=False))
    else:
        print("  (none above threshold)")

    print("\n=== Analysis 2: Annotated MSE curves ===")
    sgd = make_mse_curves(data)

    print("\n=== Done ===")
