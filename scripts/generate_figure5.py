#!/usr/bin/env python3
"""
Figure 5: Diagnostic Performance, Machine Learning Validation,
          and Feature Independence
=============================================================
Version: 2.0  (2026-03)
Authors: NeuroEng@Usach

Panels
------
A–D  Individual AUC comparison per center (Complexity vs 5 comparators):
       A = CETRAM  |  B = Cruces  |  C = Nagoya 07-11h  |  D = Nagoya 16-20h
E    LOCO AUC per center × model (LogReg / RF / SVM), DL reference
F    RF feature importance (pooled data, full fit)
G    Best handcrafted model vs end-to-end 1D-ResNet (LOCO)
H    Consolidated multi-center ROC curves (best HC model)
I–K  Spearman correlation heatmaps showing feature orthogonality
       I = CETRAM  |  J = Cruces  |  K = Nagoya 16-20h

Feature set (consistent across all panels)
-------------------------------------------
  Individual AUC: Complexity (rcMSE/HR), DFA alpha1, SampEn (S1),
                  SDNN, RMSSD, pNN50
  ML features:    above 6 + Age  (Z-score normalized per center)

Nagoya data sources
-------------------
  16-20h: japan_afternoon_features.csv — all HC metrics from same window
  07-11h: japan_evolution.csv (HR/SDNN/RMSSD) + japan_recalc_metrics.csv
          (pNN50 / DFA alpha1, 24h fallback — no per-window morning file)
  Complexity / SampEn S1: japan_{morning,afternoon}_mse.csv

DL reference: data/benchmarks/loco_cv_results.csv (benchmark_dl_loco.py)
Appendix:     generate_appendix.py — cross-center generalization matrix (RF)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
BM_DIR      = os.path.join(DATA_DIR, "benchmarks")
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
OUT_DIR     = os.path.join(FIGURES_DIR, "Figure5")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature definitions ────────────────────────────────────────────────────────

# Individual AUC comparison
AUC_METRICS = ['Complexity Index (HR-Norm)', 'HRV_DFA_alpha1',
               'Sample Entropy (S1)', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50']
AUC_LABELS  = ['Complexity (HR-Norm)', 'DFA alpha1',
               'SampEn (S1)', 'SDNN', 'RMSSD', 'pNN50']
AUC_COLORS  = {
    'Complexity (HR-Norm)': '#8E44AD',
    'DFA alpha1':           '#2980B9',
    'SampEn (S1)':          '#5DADE2',
    'SDNN':                 '#27AE60',
    'RMSSD':                '#58D68D',
    'pNN50':                '#A8D8A8',
}

# Machine learning
ML_FEATURES = ['Complexity', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50',
               'HRV_DFA_alpha1', 'SampEn_S1', 'Age']
ML_LABELS   = {
    'Complexity':     'Complexity\n(rcMSE/HR)',
    'HRV_SDNN':       'SDNN',
    'HRV_RMSSD':      'RMSSD',
    'HRV_pNN50':      'pNN50',
    'HRV_DFA_alpha1': 'DFA alpha1',
    'SampEn_S1':      'SampEn (S1)',
    'Age':            'Age',
}

CENTERS_ML    = ['CETRAM', 'Cruces', 'Nagoya']
CENTER_COLORS = {'CETRAM': '#1565C0', 'Cruces': '#2E7D32', 'Nagoya': '#E65100'}
MODELS = {
    'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced'),
    'RF':     RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                     random_state=42),
    'SVM':    SVC(probability=True, class_weight='balanced', random_state=42),
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_datasets():
    """
    Load all centers with the consolidated feature set.

    Returns
    -------
    datasets : dict keyed by display name ('CETRAM', 'Cruces',
               'Nagoya (07-11h)', 'Nagoya (16-20h)').
               Each DataFrame contains AUC_METRICS columns + Group + Site.
    df_ml    : pooled DataFrame for LOCO (CETRAM + Cruces + Nagoya 16-20h)
               with ML_FEATURES + Group / Site / Label columns.
    """
    datasets  = {}
    ml_frames = []

    # ── CETRAM ────────────────────────────────────────────────────────────────
    mse_c = pd.read_csv(os.path.join(DATA_DIR, "chile_mse.csv"))
    met_c = pd.read_csv(os.path.join(DATA_DIR, "chile_metrics.csv"))
    dem_c = pd.read_csv(os.path.join(DATA_DIR, "chile_demographics.csv"))
    dem_c = dem_c.rename(columns={'Anon_ID': 'Subject'})
    for df in (mse_c, met_c):
        df['Group'] = df['Group'].str.lower().replace(
            {'pd': 'PD', 'control': 'Control', 'parkinson': 'PD'})
    comp_c = mse_c[mse_c.Scales.isin(range(1, 6))].groupby('Subject').MSE.mean().reset_index()
    hr_c   = 60000.0 / met_c.set_index('Subject')['HRV_MeanNN']
    df_c   = comp_c.merge(
        met_c[['Subject', 'Group', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_DFA_alpha1']],
        on='Subject')
    df_c['HR']                        = df_c['Subject'].map(hr_c)
    df_c['Complexity Index (HR-Norm)'] = df_c['MSE'] / df_c['HR']
    df_c['Sample Entropy (S1)']        = df_c['Subject'].map(
        mse_c[mse_c.Scales == 1].set_index('Subject')['MSE'])
    df_c = df_c.merge(dem_c[['Subject', 'Age']], on='Subject', how='left')
    df_c['Site'] = 'CETRAM'
    datasets['CETRAM'] = df_c
    ml_frames.append(df_c)

    # ── Cruces ────────────────────────────────────────────────────────────────
    mse_s = pd.read_csv(os.path.join(DATA_DIR, "spain_mse.csv"))
    met_s = pd.read_csv(os.path.join(DATA_DIR, "spain_metrics.csv"))
    dem_s = pd.read_csv(os.path.join(DATA_DIR, "spain_demographics.csv"))
    for df in (mse_s, met_s):
        df['Group'] = df['Group'].str.lower().replace(
            {'pd': 'PD', 'control': 'Control', 'parkinson': 'PD', 'other': 'Control'})
    comp_s = mse_s[mse_s.Scales.isin(range(1, 6))].groupby('Subject').MSE.mean().reset_index()
    hr_s   = 60000.0 / met_s.set_index('Subject')['HRV_MeanNN']
    df_s   = comp_s.merge(
        met_s[['Subject', 'Group', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_DFA_alpha1']],
        on='Subject')
    df_s['HR']                        = df_s['Subject'].map(hr_s)
    df_s['Complexity Index (HR-Norm)'] = df_s['MSE'] / df_s['HR']
    df_s['Sample Entropy (S1)']        = df_s['Subject'].map(
        mse_s[mse_s.Scales == 1].groupby('Subject').MSE.mean())
    df_s = df_s.merge(dem_s[['Subject', 'Age']], on='Subject', how='left')
    df_s['Site'] = 'Cruces'
    datasets['Cruces'] = df_s
    ml_frames.append(df_s)

    # ── Nagoya ────────────────────────────────────────────────────────────────
    df_evo    = pd.read_csv(os.path.join(DATA_DIR, "japan_evolution.csv"))
    df_recalc = pd.read_csv(os.path.join(DATA_DIR, "japan_recalc_metrics.csv"))
    df_feat   = pd.read_csv(os.path.join(DATA_DIR, "japan_afternoon_features.csv"))
    df_feat['Group'] = df_feat['Group'].str.lower().replace(
        {'pd': 'PD', 'control': 'Control'})
    df_meta = pd.read_csv(os.path.join(DATA_DIR, "japan_metadata.csv"))
    if 'Subject_ID' in df_meta.columns:
        df_meta = df_meta.rename(columns={'Subject_ID': 'Subject'})

    for win_name, win_h, mse_file, is_aft in [
        ('Nagoya (07-11h)', 7,  'japan_morning_mse.csv',   False),
        ('Nagoya (16-20h)', 16, 'japan_afternoon_mse.csv', True),
    ]:
        mse_j  = pd.read_csv(os.path.join(DATA_DIR, mse_file))
        comp_j = mse_j[mse_j.Scales.isin(range(1, 21))].groupby('Subject').MSE.mean().reset_index()

        if is_aft:
            # All HC features from dedicated 16-20h window file
            met_j = df_feat[['Subject', 'Group', 'Age', 'HR',
                              'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'DFA_alpha1']].copy()
            met_j = met_j.rename(columns={'DFA_alpha1': 'HRV_DFA_alpha1'})
        else:
            # 07-11h: window HR/SDNN/RMSSD + 24h fallback for pNN50/DFA_alpha1
            win_j = df_evo[df_evo['Window_start_h'] == win_h][
                ['Subject', 'Group', 'HR', 'SDNN', 'RMSSD']].copy()
            win_j['Group'] = win_j['Group'].str.lower().replace(
                {'pd': 'PD', 'control': 'Control'})
            rec_j = df_recalc[['Subject', 'HRV_pNN50', 'DFA_alpha1']].rename(
                columns={'DFA_alpha1': 'HRV_DFA_alpha1'})
            met_j = win_j.merge(rec_j, on='Subject', how='left')
            met_j = met_j.merge(df_meta[['Subject', 'Age']], on='Subject', how='left')
            met_j = met_j.rename(columns={'SDNN': 'HRV_SDNN', 'RMSSD': 'HRV_RMSSD'})

        df_j = comp_j.merge(met_j, on='Subject')
        df_j['Complexity Index (HR-Norm)'] = df_j['MSE'] / df_j['HR']
        df_j['Sample Entropy (S1)']         = df_j['Subject'].map(
            mse_j[mse_j.Scales == 1].set_index('Subject')['MSE'])
        df_j['Site'] = 'Nagoya'
        datasets[win_name] = df_j

        if is_aft:
            ml_frames.append(df_j)

    # ── Pooled ML DataFrame ───────────────────────────────────────────────────
    col_rename = {
        'Complexity Index (HR-Norm)': 'Complexity',
        'Sample Entropy (S1)':        'SampEn_S1',
    }
    df_ml = pd.concat([d.rename(columns=col_rename) for d in ml_frames], ignore_index=True)
    df_ml['Label'] = (df_ml['Group'] == 'PD').astype(int)
    df_ml = df_ml.dropna(subset=ML_FEATURES + ['Group'])
    print(f"ML pool: {len(df_ml)} subjects  "
          + "  ".join(f"{s}={len(df_ml[df_ml.Site==s])}" for s in CENTERS_ML))

    return datasets, df_ml


# ── ML helpers ─────────────────────────────────────────────────────────────────

def normalize_per_site(df):
    df_n = df.copy()
    for site in df['Site'].unique():
        mask = df_n['Site'] == site
        df_n.loc[mask, ML_FEATURES] = StandardScaler().fit_transform(
            df_n.loc[mask, ML_FEATURES])
    return df_n


def run_loco(df_n):
    """LOCO CV for all models. Returns loco_results dict and RF feature importances."""
    X      = df_n[ML_FEATURES]
    y      = df_n['Label']
    groups = df_n['Site']
    logo   = LeaveOneGroupOut()

    loco_results = {name: {} for name in MODELS}
    for name, clf in MODELS.items():
        for train_idx, test_idx in logo.split(X, y, groups):
            site  = groups.iloc[test_idx].unique()[0]
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            probs = clf.predict_proba(X.iloc[test_idx])[:, 1]
            auc   = roc_auc_score(y.iloc[test_idx], probs)
            if auc < 0.5:
                probs = 1 - probs
                auc   = roc_auc_score(y.iloc[test_idx], probs)
            fpr, tpr, _ = roc_curve(y.iloc[test_idx], probs)
            loco_results[name][site] = {'auc': auc, 'fpr': fpr, 'tpr': tpr}

    rf_full = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_full.fit(X, y)
    importances = pd.Series(rf_full.feature_importances_,
                            index=ML_FEATURES).sort_values(ascending=False)
    return loco_results, importances


def get_auc(df, metric, group_col='Group', pos_label='PD'):
    try:
        sub    = df[[group_col, metric]].dropna()
        y_true = (sub[group_col].str.lower() == pos_label.lower()).astype(int)
        auc    = roc_auc_score(y_true, sub[metric])
        return auc if auc >= 0.5 else 1 - auc
    except Exception:
        return np.nan


def add_panel_label(ax, label, fontsize=22):
    ax.text(-0.08, 1.10, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='bottom', ha='right')


# ── Figure ─────────────────────────────────────────────────────────────────────

def generate_figure5():
    sns.set_style('ticks')

    print("Loading data...")
    datasets, df_ml = load_datasets()
    df_n = normalize_per_site(df_ml)

    print("Running LOCO cross-validation (7 features)...")
    loco_results, importances = run_loco(df_n)

    dl_ref = pd.read_csv(os.path.join(BM_DIR, "loco_cv_results.csv"))
    dl_ref['Site'] = dl_ref['Excluded'].map(
        {'Chile': 'CETRAM', 'Spain': 'Cruces', 'Japan': 'Nagoya'})
    dl_aucs = dl_ref.set_index('Site')['AUC'].to_dict()

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 36))
    gs  = fig.add_gridspec(4, 4, hspace=0.52, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])   # CETRAM AUC bars
    ax_b = fig.add_subplot(gs[0, 1])   # Cruces AUC bars
    ax_c = fig.add_subplot(gs[0, 2])   # Nagoya 07-11h AUC bars
    ax_d = fig.add_subplot(gs[0, 3])   # Nagoya 16-20h AUC bars
    ax_e = fig.add_subplot(gs[1, :2])  # LOCO × model
    ax_f = fig.add_subplot(gs[1, 2:])  # Feature importance
    ax_g = fig.add_subplot(gs[2, :2])  # HC vs DL
    ax_h = fig.add_subplot(gs[2, 2:])  # ROC curves
    ax_i = fig.add_subplot(gs[3, 0])   # CETRAM heatmap
    ax_j = fig.add_subplot(gs[3, 1])   # Cruces heatmap
    ax_k = fig.add_subplot(gs[3, 2:])  # Nagoya heatmap + colorbar

    # ── Panels A–D: Individual AUC bars ───────────────────────────────────────
    panel_order = ['CETRAM', 'Cruces', 'Nagoya (07-11h)', 'Nagoya (16-20h)']
    axes_auc    = [ax_a, ax_b, ax_c, ax_d]

    for ax, ds_key, letter in zip(axes_auc, panel_order, ['A', 'B', 'C', 'D']):
        add_panel_label(ax, letter)
        aucs = [get_auc(datasets[ds_key], m) for m in AUC_METRICS]
        valid = [(a, AUC_LABELS[j]) for j, a in enumerate(aucs) if not np.isnan(a)]
        valid.sort(key=lambda x: x[0], reverse=True)
        sorted_aucs   = [v[0] for v in valid]
        sorted_labels = [v[1] for v in valid]
        color_map     = {l: AUC_COLORS.get(l, '#95A5A6') for l in sorted_labels}

        sns.barplot(x=sorted_aucs, y=sorted_labels, hue=sorted_labels,
                    palette=color_map, legend=False, ax=ax)
        for lbl in ax.get_yticklabels():
            if 'Complexity' in lbl.get_text():
                lbl.set_fontweight('bold')
                lbl.set_fontsize(13)
                lbl.set_color('#8E44AD')

        ax.set_title(ds_key, fontsize=16, fontweight='bold')
        ax.set_xlim(0.4, 1.0)
        ax.axvline(0.5, color='black', ls='--', alpha=0.5)
        ax.set_xlabel('AUC', fontsize=11)
        ax.set_ylabel('')
        sns.despine(ax=ax)

    # ── Panel E: LOCO AUC × model ─────────────────────────────────────────────
    add_panel_label(ax_e, 'E')
    rows_e = [{'Model': m, 'Center': s, 'AUC': loco_results[m][s]['auc']}
              for m in MODELS for s in CENTERS_ML]
    df_e = pd.DataFrame(rows_e)

    model_palette = {'LogReg': '#2980B9', 'RF': '#27AE60', 'SVM': '#E67E22'}
    x_pos = np.arange(len(CENTERS_ML))
    width = 0.25
    for k, (mname, color) in enumerate(model_palette.items()):
        vals   = [df_e[(df_e.Model == mname) & (df_e.Center == c)]['AUC'].values[0]
                  for c in CENTERS_ML]
        offset = (k - 1) * width
        bars_e = ax_e.bar(x_pos + offset, vals, width, label=mname,
                          color=color, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars_e, vals):
            ax_e.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for i, site in enumerate(CENTERS_ML):
        if site in dl_aucs:
            ax_e.plot([i - 0.42, i + 0.42], [dl_aucs[site]] * 2,
                      color='#C0392B', lw=2, ls='--', zorder=5,
                      label='DL (ResNet)' if i == 0 else '_nolegend_')

    ax_e.set_xticks(x_pos)
    ax_e.set_xticklabels(CENTERS_ML, fontsize=12)
    ax_e.set_ylim(0.45, 1.02)
    ax_e.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax_e.set_ylabel('LOCO AUC', fontsize=12)
    ax_e.set_title('Handcrafted Models — LOCO AUC per Center', fontsize=16, fontweight='bold')
    ax_e.legend(fontsize=10, loc='upper left')
    sns.despine(ax=ax_e)

    # ── Panel F: Feature importance ────────────────────────────────────────────
    add_panel_label(ax_f, 'F')
    labels_fi = [ML_LABELS.get(f, f) for f in importances.index]
    colors_fi = ['#8E44AD' if 'Complexity' in l else '#5D6D7E' for l in labels_fi]
    bars_f = ax_f.barh(labels_fi[::-1], importances.values[::-1],
                       color=colors_fi[::-1], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars_f, importances.values[::-1]):
        ax_f.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                  f'{val:.3f}', va='center', fontsize=10)
    ax_f.set_xlabel('Mean Decrease in Gini Impurity', fontsize=12)
    ax_f.set_title('RF Feature Importance (pooled, full fit)', fontsize=16, fontweight='bold')
    ax_f.set_xlim(0, importances.values.max() * 1.25)
    sns.despine(ax=ax_f)

    # ── Panel G: Best HC vs DL ────────────────────────────────────────────────
    add_panel_label(ax_g, 'G')
    best_hc  = {s: max(loco_results[m][s]['auc'] for m in MODELS) for s in CENTERS_ML}
    x_g      = np.arange(len(CENTERS_ML))
    w_g      = 0.35
    bars_hc  = ax_g.bar(x_g - w_g / 2, [best_hc[s] for s in CENTERS_ML], w_g,
                        label='Handcrafted (best model)',
                        color='#8E44AD', edgecolor='white')
    bars_dl  = ax_g.bar(x_g + w_g / 2,
                        [dl_aucs.get(s, np.nan) for s in CENTERS_ML], w_g,
                        label='Deep Learning (1D-ResNet)',
                        color='#C0392B', edgecolor='white')
    for bar in list(bars_hc) + list(bars_dl):
        h = bar.get_height()
        if h > 0:
            ax_g.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                      f'{h:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax_g.set_xticks(x_g)
    ax_g.set_xticklabels(CENTERS_ML, fontsize=12)
    ax_g.set_ylim(0.45, 1.0)
    ax_g.axhline(0.5, color='gray', ls=':', alpha=0.5, label='Chance')
    ax_g.set_ylabel('LOCO AUC', fontsize=12)
    ax_g.set_title('Handcrafted Complexity Features vs End-to-End DL',
                   fontsize=16, fontweight='bold')
    ax_g.legend(fontsize=10)
    ax_g.text(0.98, 0.03, 'DL: 1D-ResNet on raw RRi\nHC: rcMSE/HR + 6 features (LOCO)',
              ha='right', va='bottom', transform=ax_g.transAxes,
              fontsize=8, color='#555', style='italic')
    sns.despine(ax=ax_g)

    # ── Panel H: Multi-center ROC curves ──────────────────────────────────────
    add_panel_label(ax_h, 'H')
    mean_aucs  = {m: np.mean([loco_results[m][s]['auc'] for s in CENTERS_ML]) for m in MODELS}
    best_model = max(mean_aucs, key=mean_aucs.get)
    roc_colors = ['#2E86AB', '#A23B72', '#F18F01']
    for site, color in zip(CENTERS_ML, roc_colors):
        res = loco_results[best_model][site]
        ax_h.plot(res['fpr'], res['tpr'],
                  label=f"{site}  (AUC = {res['auc']:.2f})",
                  color=color, linewidth=2.5)
    ax_h.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax_h.set_xlabel('False Positive Rate', fontsize=12)
    ax_h.set_ylabel('True Positive Rate', fontsize=12)
    ax_h.set_title(f'Multi-Center ROC — {best_model} (LOCO)', fontsize=16, fontweight='bold')
    ax_h.legend(fontsize=11, loc='lower right')
    sns.despine(ax=ax_h)

    # ── Panels I–K: Feature orthogonality heatmaps ────────────────────────────
    hm_centers = ['CETRAM', 'Cruces', 'Nagoya (16-20h)']
    hm_axes    = [ax_i, ax_j, ax_k]

    for ax, center, letter in zip(hm_axes, hm_centers, ['I', 'J', 'K']):
        add_panel_label(ax, letter)
        corr_df = datasets[center][AUC_METRICS].copy()
        corr_df.columns = AUC_LABELS
        corr_matrix = corr_df.corr(method='spearman')
        show_cbar = (center == 'Nagoya (16-20h)')
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, ax=ax, fmt='.2f',
                    cbar=show_cbar, annot_kws={'size': 9})
        n = len(datasets[center])
        ax.set_title(f'Feature Orthogonality\n{center} (n={n})',
                     fontsize=14, fontweight='bold')
        if center != 'CETRAM':
            ax.set_yticklabels([])

    # ── Suptitle & save ────────────────────────────────────────────────────────
    n_info = "  |  ".join(f"{s}: n={len(df_ml[df_ml.Site==s])}" for s in CENTERS_ML)
    plt.suptitle(
        'Figure 5: Diagnostic Performance, Machine Learning Validation, '
        'and Feature Independence\n'
        f'{n_info}  |  7 features  |  LOCO cross-validation  |  Z-scored per center',
        fontsize=20, fontweight='bold', y=1.0)
    fig.subplots_adjust(top=0.965, bottom=0.03, left=0.07, right=0.97,
                        hspace=0.52, wspace=0.35)

    out_path = os.path.join(OUT_DIR, 'Figure5.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"Figure 5 saved to {out_path}")


if __name__ == '__main__':
    generate_figure5()
