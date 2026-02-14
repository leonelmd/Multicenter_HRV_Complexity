#!/usr/bin/env python3
"""
Figure 5: Comparative Diagnostic Performance and Feature Independence
- AUC Comparison across centers (Complexity vs Standard Metrics)
- Multi-center ROC curves for the primary biomarker
- Correlation analysis showing the distinct nature of fractal complexity
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def add_panel_label(ax, label):
    ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=24, fontweight='bold', va='bottom', ha='right')

def get_auc(df, metric, group_col='Group', pos_label='PD'):
    try:
        y_true = (df[group_col].str.lower() == pos_label.lower()).astype(int)
        y_score = df[metric]
        auc = roc_auc_score(y_true, y_score)
        if auc < 0.5: auc = 1 - auc
        return auc
    except:
        return np.nan

def generate_figure5():
    sns.set_style("ticks")
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    datasets = {}
    
    # 1. LOAD AND PREPARE DATA
    # --- CHILE ---
    df_c_mse = pd.read_csv(os.path.join(DATA_DIR, "chile_mse.csv"))
    df_c_met = pd.read_csv(os.path.join(DATA_DIR, "chile_metrics.csv"))
    df_c_mse['Group'] = df_c_mse['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD'})
    df_c_met['Group'] = df_c_met['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD'})
    comp_c = df_c_mse[df_c_mse.Scales.isin(range(1,6))].groupby('Subject').MSE.mean().reset_index()
    hr_c = 60000.0 / df_c_met.set_index('Subject')['HRV_MeanNN']
    df_c = pd.merge(comp_c, df_c_met[['Subject','Group','HRV_SDNN','HRV_RMSSD','HRV_DFA_alpha1']], on='Subject')
    df_c['HR'] = df_c['Subject'].map(hr_c)
    df_c['Complexity Index (HR-Norm)'] = df_c['MSE'] / df_c['HR']
    df_c['Sample Entropy (S1)'] = df_c['Subject'].map(df_c_mse[df_c_mse.Scales==1].set_index('Subject')['MSE'].to_dict())
    datasets['Chile'] = df_c

    # --- SPAIN ---
    df_s_mse = pd.read_csv(os.path.join(DATA_DIR, "spain_mse.csv"))
    df_s_met = pd.read_csv(os.path.join(DATA_DIR, "spain_metrics.csv"))
    df_s_mse['Group'] = df_s_mse['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD','other':'Control'})
    df_s_met['Group'] = df_s_met['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD','other':'Control'})
    comp_s = df_s_mse[df_s_mse.Scales.isin(range(1,6))].groupby('Subject').MSE.mean().reset_index()
    hr_s = 60000.0 / df_s_met.set_index('Subject')['HRV_MeanNN']
    df_s = pd.merge(comp_s, df_s_met[['Subject','Group','HRV_SDNN','HRV_RMSSD','HRV_DFA_alpha1']], on='Subject')
    df_s['HR'] = df_s['Subject'].map(hr_s)
    df_s['Complexity Index (HR-Norm)'] = df_s['MSE'] / df_s['HR']
    df_s['Sample Entropy (S1)'] = df_s['Subject'].map(df_s_mse[df_s_mse.Scales==1].groupby('Subject').MSE.mean().to_dict())
    datasets['Spain'] = df_s

    # --- JAPAN ---
    df_j_evo = pd.read_csv(os.path.join(DATA_DIR, "japan_evolution.csv"))
    df_j_meta = pd.read_csv(os.path.join(DATA_DIR, "japan_metadata.csv"))
    if 'Subject_ID' in df_j_meta.columns: df_j_meta = df_j_meta.rename(columns={'Subject_ID': 'Subject'})
    def parse_t(s): 
        try: parts = str(s).split(':'); return int(parts[0]) + int(parts[1])/60.0 + int(parts[2])/3600.0
        except: return np.nan
    start_map = dict(zip(df_j_meta['Subject'], df_j_meta['Start_Time'].apply(parse_t)))
    df_j_evo['Clock_T'] = (df_j_evo['Subject'].map(start_map) + df_j_evo['Time_h']) % 24

    for win_name, (start_h, end_h), mse_file in [
        ('Nagoya (7-11 AM)', (7, 11), 'japan_morning_mse.csv'),
        ('Nagoya (1-5 PM)', (13, 17), 'japan_afternoon_mse.csv')
    ]:
        df_j_mse = pd.read_csv(os.path.join(DATA_DIR, mse_file))
        mask = (df_j_evo['Clock_T'] >= start_h) & (df_j_evo['Clock_T'] < end_h)
        df_j_met = df_j_evo[mask].groupby(['Subject','Group'])[['HR','SDNN','RMSSD','Alpha1']].mean().reset_index()
        df_j_met['Group'] = df_j_met['Group'].str.lower().replace({'pd':'PD','control':'Control'})
        comp_j = df_j_mse[df_j_mse.Scales.isin(range(1,21))].groupby('Subject').MSE.mean().reset_index()
        df_j = pd.merge(comp_j, df_j_met, on='Subject')
        df_j['Complexity Index (HR-Norm)'] = df_j['MSE'] / df_j['HR']
        df_j['Sample Entropy (S1)'] = df_j['Subject'].map(df_j_mse[df_j_mse.Scales==1].set_index('Subject')['MSE'].to_dict())
        df_j = df_j.rename(columns={'SDNN':'HRV_SDNN', 'RMSSD':'HRV_RMSSD', 'Alpha1':'HRV_DFA_alpha1'})
        datasets[win_name] = df_j

    # 2. AUC BAR CHARTS (4 Panels)
    metrics_to_compare = ['Complexity Index (HR-Norm)', 'HRV_DFA_alpha1', 'Sample Entropy (S1)', 'HRV_SDNN', 'HRV_RMSSD']
    metric_labels = ['Complexity (HR-Norm)', 'DFA Alpha 1', 'SampEn (S1)', 'SDNN', 'RMSSD']
    panel_order = ['Chile', 'Spain', 'Nagoya (7-11 AM)', 'Nagoya (1-5 PM)']
    
    for i, ds_key in enumerate(panel_order):
        ax = fig.add_subplot(gs[0 if i < 3 else 1, i % 3])
        add_panel_label(ax, chr(65 + i))
        aucs = [get_auc(datasets[ds_key], m) for m in metrics_to_compare]
        sorted_idx = np.argsort(aucs)[::-1]
        sorted_aucs = [aucs[idx] for idx in sorted_idx]
        sorted_labels = [metric_labels[idx] for idx in sorted_idx]
        colors = ['#D62828' if 'Complexity' in l else '#457B9D' for l in sorted_labels]
        sns.barplot(x=sorted_aucs, y=sorted_labels, palette=colors, ax=ax)
        ax.set_title(f"{ds_key}", fontsize=18, fontweight='bold')
        ax.set_xlim(0.4, 1.0); ax.axvline(0.5, color='black', ls='--', alpha=0.5)
        ax.set_xlabel("AUC")

    # 3. CONSOLIDATED ROC CURVES (Panel E)
    ax_roc = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_roc, 'E')
    roc_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C77DFF']
    for ds_key, color in zip(panel_order, roc_colors):
        df = datasets[ds_key]
        y_true = (df['Group'].str.lower() == 'pd').astype(int)
        scores = df['Complexity Index (HR-Norm)']
        if roc_auc_score(y_true, scores) < 0.5: scores = -scores
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = get_auc(df, 'Complexity Index (HR-Norm)')
        ax_roc.plot(fpr, tpr, label=f"{ds_key} ({auc:.2f})", color=color, linewidth=3)
        
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_title("Multi-center Primary ROC", fontsize=18, fontweight='bold')
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.legend(loc='lower right', fontsize=10)

    # 4. CORRELATION HEATMAP (Panel F)
    ax_corr = fig.add_subplot(gs[1, 2])
    add_panel_label(ax_corr, 'F')
    corr_df = datasets['Spain'][metrics_to_compare]
    corr_df.columns = metric_labels
    corr_matrix = corr_df.corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax_corr, fmt='.2f')
    ax_corr.set_title("Feature Orthogonality (Spain)", fontsize=18, fontweight='bold')

    plt.suptitle("Figure 5: Multi-center Diagnostic Performance and Biomarker Independence", fontsize=28, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, "Figure5", "Figure5.png")
    plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight'), dpi=300)
    print("Figure 5 finalized.")

if __name__ == "__main__":
    generate_figure5()
