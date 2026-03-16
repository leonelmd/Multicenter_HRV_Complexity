#!/usr/bin/env python3
"""
Figure 7: Machine Learning Validation — Handcrafted Complexity Features vs Deep Learning
(Consolidates former Figs 7 & 8)

Panel A — RF feature importance (pooled data, full fit)
Panel B — LOCO AUC per center × model (LogReg / RF / SVM)
Panel C — Handcrafted vs end-to-end DL: per-center AUC comparison
Panel D — Cross-center generalization matrix (train-on-one, test-on-another; RF)
Panel E — ROC curves per center (best handcrafted model, LOCO)
Panel F — Primary biomarker by group × center (Complexity = rcMSE/HR, Z-scored)

Features: Complexity (rcMSE nAUC / HR), SDNN, RMSSD, pNN50, DFA_alpha1, Age
          Z-score normalized per center (corrects for protocol/hardware differences)
DL reference (Panels C): 1D-ResNet LOCO cross-validation from benchmarks/loco_cv_results.csv
"""

import os
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
from scipy.stats import mannwhitneyu

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
BM_DIR      = os.path.join(DATA_DIR, "benchmarks")
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
OUT_DIR     = os.path.join(FIGURES_DIR, "Figure7")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS   = {'Control': '#2E86AB', 'PD': '#D62828'}
CENTERS  = ['CETRAM', 'Cruces', 'Nagoya']
CENTER_COLORS = {'CETRAM': '#1565C0', 'Cruces': '#2E7D32', 'Nagoya': '#E65100'}
MODELS   = {'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced'),
            'RF':     RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
            'SVM':    SVC(probability=True, class_weight='balanced', random_state=42)}

def add_panel_label(ax, label, fontsize=22):
    ax.text(-0.08, 1.10, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='bottom', ha='right')

def get_sig(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    """Return pooled DataFrame with features, Group, Site."""
    # CETRAM
    mse_c = pd.read_csv(os.path.join(DATA_DIR, "chile_mse.csv"))
    met_c = pd.read_csv(os.path.join(DATA_DIR, "chile_metrics.csv"))
    mse_c['Group'] = mse_c['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD'})
    met_c['Group'] = met_c['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD'})
    comp_c = mse_c[mse_c.Scales.isin(range(1,6))].groupby('Subject')['MSE'].mean().reset_index()
    hr_c   = 60000.0 / met_c.set_index('Subject')['HRV_MeanNN']
    df_c   = comp_c.merge(met_c[['Subject','Group','HRV_SDNN','HRV_RMSSD','HRV_pNN50','HRV_DFA_alpha1']], on='Subject')
    df_c['HR']         = df_c['Subject'].map(hr_c)
    df_c['Complexity'] = df_c['MSE'] / df_c['HR']
    dem_c  = pd.read_csv(os.path.join(DATA_DIR, "chile_demographics.csv"))
    df_c   = df_c.merge(dem_c[['Anon_ID','Age']], left_on='Subject', right_on='Anon_ID', how='left')
    df_c['Site'] = 'CETRAM'

    # Cruces
    mse_s = pd.read_csv(os.path.join(DATA_DIR, "spain_mse.csv"))
    met_s = pd.read_csv(os.path.join(DATA_DIR, "spain_metrics.csv"))
    mse_s['Group'] = mse_s['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD','other':'Control'})
    met_s['Group'] = met_s['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD','other':'Control'})
    comp_s = mse_s[mse_s.Scales.isin(range(1,6))].groupby('Subject')['MSE'].mean().reset_index()
    hr_s   = 60000.0 / met_s.set_index('Subject')['HRV_MeanNN']
    df_s   = comp_s.merge(met_s[['Subject','Group','HRV_SDNN','HRV_RMSSD','HRV_pNN50','HRV_DFA_alpha1']], on='Subject')
    df_s['HR']         = df_s['Subject'].map(hr_s)
    df_s['Complexity'] = df_s['MSE'] / df_s['HR']
    dem_s  = pd.read_csv(os.path.join(DATA_DIR, "spain_demographics.csv"))
    df_s   = df_s.merge(dem_s[['Subject','Age']], on='Subject', how='left')
    df_s['Site'] = 'Cruces'

    # Nagoya (16-20h window, scales 1-20)
    mse_j  = pd.read_csv(os.path.join(DATA_DIR, "japan_afternoon_mse.csv"))
    rec_j  = pd.read_csv(os.path.join(DATA_DIR, "japan_recalc_metrics.csv"))
    rec_j['HR'] = 60000.0 / rec_j['HRV_MeanNN']
    comp_j = mse_j[mse_j.Scales.isin(range(1,21))].groupby('Subject')['MSE'].mean().reset_index()
    df_j   = comp_j.merge(rec_j[['Subject','Group','HR','HRV_SDNN','HRV_RMSSD','HRV_pNN50','DFA_alpha1']], on='Subject')
    df_j.rename(columns={'DFA_alpha1': 'HRV_DFA_alpha1'}, inplace=True)
    df_j['Complexity'] = df_j['MSE'] / df_j['HR']
    dem_j  = pd.read_csv(os.path.join(DATA_DIR, "japan_metadata.csv"))
    df_j   = df_j.merge(dem_j[['Subject_ID','Age']], left_on='Subject', right_on='Subject_ID', how='left')
    df_j['Group'] = df_j['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD'})
    df_j['Site']  = 'Nagoya'

    df = pd.concat([df_c, df_s, df_j], ignore_index=True)
    df['Label'] = (df['Group'] == 'PD').astype(int)
    features = ['Complexity', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_DFA_alpha1', 'Age']
    df = df.dropna(subset=features + ['Group'])
    print(f"Loaded: {len(df)} subjects  "
          + "  ".join(f"{s}={len(df[df.Site==s])}" for s in CENTERS))
    return df, features


def normalize_per_site(df, features):
    """Z-score normalize each feature within each center."""
    df_n = df.copy()
    for site in CENTERS:
        mask = df_n['Site'] == site
        df_n.loc[mask, features] = StandardScaler().fit_transform(df_n.loc[mask, features])
    return df_n


# ── LOCO cross-validation ──────────────────────────────────────────────────────

def run_loco(df_n, features):
    """
    Leave-One-Center-Out CV for each model.
    Returns:
      loco_results  : {model_name: {center: {'auc', 'fpr', 'tpr'}}}
      importances   : Series (RF, full-fit)
    """
    X      = df_n[features]
    y      = df_n['Label']
    groups = df_n['Site']
    logo   = LeaveOneGroupOut()

    loco_results = {name: {} for name in MODELS}

    for name, clf in MODELS.items():
        for train_idx, test_idx in logo.split(X, y, groups):
            site = groups.iloc[test_idx].unique()[0]
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            probs = clf.predict_proba(X.iloc[test_idx])[:, 1]
            auc   = roc_auc_score(y.iloc[test_idx], probs)
            if auc < 0.5:
                probs = 1 - probs
                auc   = roc_auc_score(y.iloc[test_idx], probs)
            fpr, tpr, _ = roc_curve(y.iloc[test_idx], probs)
            loco_results[name][site] = {'auc': auc, 'fpr': fpr, 'tpr': tpr}

    # RF feature importances (full fit)
    rf_full = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_full.fit(X, y)
    importances = pd.Series(rf_full.feature_importances_, index=features).sort_values(ascending=False)

    return loco_results, importances


def compute_generalization_matrix(df_n, features):
    """
    Train RF on each single center, test on each other center.
    Diagonal = LOCO AUC (from run_loco RF results, passed in).
    Returns: (3×3 AUC matrix, row=train, col=test).
    """
    X = df_n[features].values
    y = df_n['Label'].values
    sites_arr = df_n['Site'].values

    mat = np.full((3, 3), np.nan)
    rf  = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

    for i, train_site in enumerate(CENTERS):
        for j, test_site in enumerate(CENTERS):
            if train_site == test_site:
                continue
            train_mask = sites_arr == train_site
            test_mask  = sites_arr == test_site
            rf.fit(X[train_mask], y[train_mask])
            probs = rf.predict_proba(X[test_mask])[:, 1]
            auc = roc_auc_score(y[test_mask], probs)
            if auc < 0.5:
                auc = 1 - auc
            mat[i, j] = auc

    return mat


# ── Figure ─────────────────────────────────────────────────────────────────────

def generate_figure7():
    sns.set_style('ticks')

    df, features = load_data()
    df_n = normalize_per_site(df, features)

    print("Running LOCO cross-validation...")
    loco_results, importances = run_loco(df_n, features)

    print("Computing cross-center generalization matrix...")
    gen_mat = compute_generalization_matrix(df_n, features)
    # Fill diagonal with LOCO RF AUC
    for i, site in enumerate(CENTERS):
        gen_mat[i, i] = loco_results['RF'][site]['auc']

    fig = plt.figure(figsize=(24, 20))
    gs  = fig.add_gridspec(3, 2, hspace=0.48, wspace=0.32)

    feat_labels = {
        'Complexity':     'Complexity\n(rcMSE/HR)',
        'HRV_SDNN':       'SDNN',
        'HRV_RMSSD':      'RMSSD',
        'HRV_pNN50':      'pNN50',
        'HRV_DFA_alpha1': 'DFA alpha1',
        'Age':            'Age',
    }

    # ── Panel A: Feature Importance ────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, 'A')

    labels = [feat_labels.get(f, f) for f in importances.index]
    colors_fi = ['#8E44AD' if 'Complexity' in l else '#5D6D7E' for l in labels]
    bars = ax_a.barh(labels[::-1], importances.values[::-1], color=colors_fi[::-1],
                     edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, importances.values[::-1]):
        ax_a.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                  f'{val:.3f}', va='center', fontsize=10)
    ax_a.set_xlabel('Mean Decrease in Gini Impurity', fontsize=12)
    ax_a.set_title('Feature Importance (RF, pooled data)', fontsize=16, fontweight='bold')
    ax_a.set_xlim(0, importances.values.max() * 1.25)
    sns.despine(ax=ax_a)

    # ── Panel B: LOCO AUC per center × model ──────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, 'B')

    rows_b = []
    for mname, center_dict in loco_results.items():
        for site, res in center_dict.items():
            rows_b.append({'Model': mname, 'Center': site, 'AUC': res['auc']})
    df_b = pd.DataFrame(rows_b)

    model_palette = {'LogReg': '#2980B9', 'RF': '#27AE60', 'SVM': '#E67E22'}
    x_pos  = np.arange(len(CENTERS))
    n_mod  = len(MODELS)
    width  = 0.25
    for k, (mname, color) in enumerate(model_palette.items()):
        vals = [df_b[(df_b.Model == mname) & (df_b.Center == c)]['AUC'].values[0]
                for c in CENTERS]
        offset = (k - 1) * width
        bars_b = ax_b.bar(x_pos + offset, vals, width, label=mname,
                           color=color, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars_b, vals):
            ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                      f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # DL reference from benchmarks
    dl_ref = pd.read_csv(os.path.join(BM_DIR, "loco_cv_results.csv"))
    dl_ref['Site'] = dl_ref['Excluded'].map({'Chile': 'CETRAM', 'Spain': 'Cruces', 'Japan': 'Nagoya'})
    for i, site in enumerate(CENTERS):
        row = dl_ref[dl_ref['Site'] == site]
        if not row.empty:
            dl_auc = row['AUC'].values[0]
            ax_b.plot([i - 0.45, i + 0.45], [dl_auc, dl_auc],
                      color='#C0392B', lw=2, ls='--', zorder=5,
                      label='DL (ResNet)' if i == 0 else '_nolegend_')

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(CENTERS, fontsize=12)
    ax_b.set_ylim(0.45, 1.02)
    ax_b.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax_b.set_ylabel('LOCO AUC', fontsize=12)
    ax_b.set_title('Handcrafted Models — LOCO AUC per Center', fontsize=16, fontweight='bold')
    ax_b.legend(fontsize=10, loc='upper left')
    sns.despine(ax=ax_b)

    # ── Panel C: Handcrafted (best) vs DL ─────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, 'C')

    # Best HC model per center
    best_hc = {}
    for site in CENTERS:
        best_hc[site] = max(loco_results[m][site]['auc'] for m in MODELS)

    # DL from benchmark
    dl_aucs = {row['Site']: row['AUC'] for _, row in dl_ref.assign(
        Site=dl_ref['Excluded'].map({'Chile': 'CETRAM', 'Spain': 'Cruces', 'Japan': 'Nagoya'})
    ).iterrows()}

    x_c   = np.arange(len(CENTERS))
    w_c   = 0.35
    bars_hc = ax_c.bar(x_c - w_c / 2, [best_hc[s] for s in CENTERS], w_c,
                        label='Handcrafted (best HC model)',
                        color=['#8E44AD'] * 3, edgecolor='white')
    bars_dl = ax_c.bar(x_c + w_c / 2, [dl_aucs.get(s, np.nan) for s in CENTERS], w_c,
                        label='Deep Learning (1D-ResNet LOCO)',
                        color=['#C0392B'] * 3, edgecolor='white')

    for bar in list(bars_hc) + list(bars_dl):
        h = bar.get_height()
        if h > 0:
            ax_c.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                      f'{h:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax_c.set_xticks(x_c); ax_c.set_xticklabels(CENTERS, fontsize=12)
    ax_c.set_ylim(0.45, 1.0)
    ax_c.axhline(0.5, color='gray', ls=':', alpha=0.5, label='Chance')
    ax_c.set_ylabel('LOCO AUC', fontsize=12)
    ax_c.set_title('Handcrafted Complexity Features vs End-to-End DL', fontsize=16, fontweight='bold')
    ax_c.legend(fontsize=10)
    ax_c.text(0.98, 0.03, 'DL: 1D-ResNet on raw RRi\nHandcrafted: rcMSE/HR + 5 HRV features',
              ha='right', va='bottom', transform=ax_c.transAxes,
              fontsize=8, color='#555', style='italic')
    sns.despine(ax=ax_c)

    # ── Panel D: Cross-center generalization matrix ────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, 'D')

    annot = np.array([[f'{v:.2f}' for v in row] for row in gen_mat])
    # Mark diagonal differently
    for i in range(3):
        annot[i, i] = f'{gen_mat[i, i]:.2f}\n(LOCO)'

    sns.heatmap(gen_mat, annot=annot, fmt='', cmap='YlOrRd', vmin=0.45, vmax=1.0,
                xticklabels=CENTERS, yticklabels=CENTERS,
                linewidths=0.5, linecolor='white', ax=ax_d,
                cbar_kws={'label': 'AUC'})
    ax_d.set_xlabel('Test Center', fontsize=12)
    ax_d.set_ylabel('Training Center', fontsize=12)
    ax_d.set_title('Cross-Center Generalization Matrix\n(RF, handcrafted features)',
                   fontsize=16, fontweight='bold')

    # ── Panel E: ROC curves per center (best model = highest mean LOCO AUC) ───
    ax_e = fig.add_subplot(gs[2, 0])
    add_panel_label(ax_e, 'E')

    # Choose model with highest mean LOCO AUC
    mean_aucs = {m: np.mean([loco_results[m][s]['auc'] for s in CENTERS]) for m in MODELS}
    best_model = max(mean_aucs, key=mean_aucs.get)

    for site in CENTERS:
        res = loco_results[best_model][site]
        ax_e.plot(res['fpr'], res['tpr'],
                  label=f"{site}  (AUC = {res['auc']:.2f})",
                  color=CENTER_COLORS[site], lw=2.5)
    ax_e.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax_e.set_xlabel('False Positive Rate', fontsize=12)
    ax_e.set_ylabel('True Positive Rate', fontsize=12)
    ax_e.set_title(f'ROC Curves — {best_model} (LOCO)', fontsize=16, fontweight='bold')
    ax_e.legend(fontsize=11, loc='lower right')
    sns.despine(ax=ax_e)

    # ── Panel F: Complexity by group × center ─────────────────────────────────
    ax_f = fig.add_subplot(gs[2, 1])
    add_panel_label(ax_f, 'F')

    df_f = df_n[['Site', 'Group', 'Complexity']].copy()
    df_f['Group'] = df_f['Group'].str.lower().str.strip().replace({'control':'Control','pd':'PD'})
    df_f = df_f[df_f['Group'].isin(['Control', 'PD'])]

    sns.boxplot(data=df_f, x='Site', y='Complexity', hue='Group',
                palette=COLORS, hue_order=['Control', 'PD'],
                order=CENTERS, showfliers=False, width=0.55, linewidth=1.2, ax=ax_f)
    sns.stripplot(data=df_f, x='Site', y='Complexity', hue='Group',
                  palette=COLORS, hue_order=['Control', 'PD'],
                  order=CENTERS, dodge=True, alpha=0.35, size=3.5, ax=ax_f, legend=False)

    # Significance brackets
    y_top = df_f['Complexity'].quantile(0.97) * 1.05
    h     = df_f['Complexity'].std() * 0.08
    for xi, site in enumerate(CENTERS):
        sub = df_f[df_f['Site'] == site]
        c_v = sub[sub['Group'] == 'Control']['Complexity'].dropna()
        p_v = sub[sub['Group'] == 'PD']['Complexity'].dropna()
        if len(c_v) > 3 and len(p_v) > 3:
            _, pval = mannwhitneyu(c_v, p_v, alternative='two-sided')
            sig   = get_sig(pval)
            color = '#D62828' if pval < 0.05 else '#888888'
            ax_f.text(xi, y_top, sig, ha='center', va='bottom',
                      fontsize=13, fontweight='bold', color=color)

    ax_f.set_xlabel('')
    ax_f.set_ylabel('Complexity — rcMSE/HR  (Z-scored per center)', fontsize=11)
    ax_f.set_title('PD vs Control: Primary Biomarker Distribution', fontsize=16, fontweight='bold')
    handles, lbls = ax_f.get_legend_handles_labels()
    ax_f.legend(handles[:2], lbls[:2], fontsize=11, loc='upper right')
    sns.despine(ax=ax_f)

    # ── Suptitle & save ────────────────────────────────────────────────────────
    n_by_center = "  |  ".join(f"{s}: n={len(df[df.Site==s])}" for s in CENTERS)
    plt.suptitle(
        f'Figure 7: Machine Learning Validation — Handcrafted Complexity Features vs Deep Learning\n'
        f'{n_by_center}  |  Features Z-scored per center  |  LOCO cross-validation',
        fontsize=18, fontweight='bold', y=0.995)
    fig.subplots_adjust(top=0.935, hspace=0.48, wspace=0.32)

    out_path = os.path.join(OUT_DIR, 'Figure7.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"Figure 7 saved to {out_path}")


if __name__ == '__main__':
    generate_figure7()
