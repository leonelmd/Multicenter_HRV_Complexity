#!/usr/bin/env python3
"""
Appendix Figure: Cross-Center Generalization Matrix
====================================================
Version: 2.0  (2026-03)
Authors: NeuroEng@Usach

Supplementary figure showing how well models trained on a single center
generalize to unseen centers (RF, handcrafted features).

Row = training center, Column = test center.
Diagonal = LOCO AUC (train on all others, test on this center).

Feature set (consistent with Figure 5)
---------------------------------------
  Complexity (rcMSE/HR), SDNN, RMSSD, pNN50, DFA alpha1, SampEn S1, Age
  Z-score normalized per center before training.

Output
------
  figures/Appendix/FigureAppendix.png
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
OUT_DIR     = os.path.join(FIGURES_DIR, "Appendix")
os.makedirs(OUT_DIR, exist_ok=True)

CENTERS_ML = ['CETRAM', 'Cruces', 'Nagoya']
ML_FEATURES = ['Complexity', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50',
               'HRV_DFA_alpha1', 'SampEn_S1', 'Age']


# ── Data loading (mirrors generate_figure5.py) ─────────────────────────────────

def load_ml_data():
    """Load pooled ML DataFrame (CETRAM + Cruces + Nagoya 16-20h)."""
    frames = []

    # CETRAM
    mse_c = pd.read_csv(os.path.join(DATA_DIR, "chile_mse.csv"))
    met_c = pd.read_csv(os.path.join(DATA_DIR, "chile_metrics.csv"))
    dem_c = pd.read_csv(os.path.join(DATA_DIR, "chile_demographics.csv")).rename(
        columns={'Anon_ID': 'Subject'})
    for df in (mse_c, met_c):
        df['Group'] = df['Group'].str.lower().replace(
            {'pd': 'PD', 'control': 'Control', 'parkinson': 'PD'})
    comp_c = mse_c[mse_c.Scales.isin(range(1, 6))].groupby('Subject').MSE.mean().reset_index()
    hr_c   = 60000.0 / met_c.set_index('Subject')['HRV_MeanNN']
    df_c   = comp_c.merge(
        met_c[['Subject', 'Group', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_DFA_alpha1']],
        on='Subject')
    df_c['Complexity'] = df_c['MSE'] / df_c['Subject'].map(hr_c)
    df_c['SampEn_S1']  = df_c['Subject'].map(
        mse_c[mse_c.Scales == 1].set_index('Subject')['MSE'])
    df_c = df_c.merge(dem_c[['Subject', 'Age']], on='Subject', how='left')
    df_c['Site'] = 'CETRAM'
    frames.append(df_c)

    # Cruces
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
    df_s['Complexity'] = df_s['MSE'] / df_s['Subject'].map(hr_s)
    df_s['SampEn_S1']  = df_s['Subject'].map(
        mse_s[mse_s.Scales == 1].groupby('Subject').MSE.mean())
    df_s = df_s.merge(dem_s[['Subject', 'Age']], on='Subject', how='left')
    df_s['Site'] = 'Cruces'
    frames.append(df_s)

    # Nagoya 16-20h (single authoritative source)
    mse_j  = pd.read_csv(os.path.join(DATA_DIR, "japan_afternoon_mse.csv"))
    feat_j = pd.read_csv(os.path.join(DATA_DIR, "japan_afternoon_features.csv"))
    feat_j['Group'] = feat_j['Group'].str.lower().replace(
        {'pd': 'PD', 'control': 'Control'})
    comp_j = mse_j[mse_j.Scales.isin(range(1, 21))].groupby('Subject').MSE.mean().reset_index()
    df_j   = comp_j.merge(
        feat_j[['Subject', 'Group', 'Age', 'HR',
                'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'DFA_alpha1']],
        on='Subject')
    df_j = df_j.rename(columns={'DFA_alpha1': 'HRV_DFA_alpha1'})
    df_j['Complexity'] = df_j['MSE'] / df_j['HR']
    df_j['SampEn_S1']  = df_j['Subject'].map(
        mse_j[mse_j.Scales == 1].set_index('Subject')['MSE'])
    df_j['Site'] = 'Nagoya'
    frames.append(df_j)

    df_ml = pd.concat(frames, ignore_index=True)
    df_ml['Label'] = (df_ml['Group'] == 'PD').astype(int)
    df_ml = df_ml.dropna(subset=ML_FEATURES + ['Group'])
    return df_ml


def normalize_per_site(df):
    df_n = df.copy()
    for site in df['Site'].unique():
        mask = df_n['Site'] == site
        df_n.loc[mask, ML_FEATURES] = StandardScaler().fit_transform(
            df_n.loc[mask, ML_FEATURES])
    return df_n


def compute_generalization_matrix(df_n):
    """
    Train RF on each single center, test on each other center.
    Diagonal = LOCO AUC (train on other two, test on this center).
    Returns 3×3 AUC matrix (row=train, col=test).
    """
    X         = df_n[ML_FEATURES].values
    y         = df_n['Label'].values
    sites_arr = df_n['Site'].values
    mat       = np.full((3, 3), np.nan)
    rf        = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

    # Off-diagonal: train on one, test on another
    for i, train_site in enumerate(CENTERS_ML):
        for j, test_site in enumerate(CENTERS_ML):
            if train_site == test_site:
                continue
            rf.fit(X[sites_arr == train_site], y[sites_arr == train_site])
            probs = rf.predict_proba(X[sites_arr == test_site])[:, 1]
            auc   = roc_auc_score(y[sites_arr == test_site], probs)
            mat[i, j] = max(auc, 1 - auc)

    # Diagonal: LOCO (train on other two)
    logo = LeaveOneGroupOut()
    groups = df_n['Site']
    for train_idx, test_idx in logo.split(X, y, groups):
        site = groups.iloc[test_idx].unique()[0]
        j    = CENTERS_ML.index(site)
        rf.fit(X[train_idx], y[train_idx])
        probs = rf.predict_proba(X[test_idx])[:, 1]
        auc   = roc_auc_score(y[test_idx], probs)
        mat[j, j] = max(auc, 1 - auc)

    return mat


def generate_appendix():
    sns.set_style('ticks')

    print("Loading data...")
    df_ml = load_ml_data()
    df_n  = normalize_per_site(df_ml)
    print(f"  {len(df_ml)} subjects  "
          + "  ".join(f"{s}={len(df_ml[df_ml.Site==s])}" for s in CENTERS_ML))

    print("Computing generalization matrix...")
    mat = compute_generalization_matrix(df_n)

    fig, ax = plt.subplots(figsize=(7, 6))

    annot = np.array([[f'{v:.2f}' for v in row] for row in mat])
    for j in range(3):
        annot[j, j] = f'{mat[j, j]:.2f}\n(LOCO)'

    sns.heatmap(mat, annot=annot, fmt='', cmap='YlOrRd', vmin=0.45, vmax=1.0,
                xticklabels=CENTERS_ML, yticklabels=CENTERS_ML,
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'AUC'})
    ax.set_xlabel('Test Center', fontsize=13)
    ax.set_ylabel('Training Center', fontsize=13)
    ax.set_title(
        'Appendix: Cross-Center Generalization Matrix\n'
        'RF, 7 handcrafted features, Z-scored per center',
        fontsize=14, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'FigureAppendix.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"Appendix figure saved to {out_path}")


def generate_figure7():
    generate_appendix()


if __name__ == '__main__':
    generate_appendix()
