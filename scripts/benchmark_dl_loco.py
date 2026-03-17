#!/usr/bin/env python3
"""
DL Benchmark: 1D ResNet LOCO Cross-Validation
==============================================
Version: 2.0  (2026-03)
Authors: NeuroEng@Usach

Trains a 1D ResNet on raw RRi sequences using Leave-One-Center-Out (LOCO)
cross-validation and saves the per-fold AUC to data/benchmarks/loco_cv_results.csv.

Architecture
------------
  1D ResNet: Conv1d stem → 2 residual blocks → global average pooling → FC(64→32→1)
  Input: fixed-length 400-beat windows, zero-centred at 0.9 s mean RRi

Data handling
-------------
  - All three centers use sliding windows of SEQ_LEN=400 beats, STRIDE=50
  - Nagoya filtered to 16:00–20:00 clock time (best PD-discrimination window,
    consistent with handcrafted features in the appendix figure)
  - Class imbalance corrected with WeightedRandomSampler
  - Subject-level prediction = mean probability over all windows (voting)

Training
--------
  40 epochs, Adam lr=0.001, BCEWithLogitsLoss, batch size 128
  Runs on Apple MPS if available, else CPU

Raw data paths (NOT included in public_release — available on request)
-----------------------------------------------------------------------
  CETRAM_RRI_DIR : folder with PD/ and Control/ subfolders, each containing
                   *_cleaned.csv files with a 'sample' column (1000 Hz indices).
  CRUCES_RRI_DIR : folder with per-subject *.csv files (one column, RRi in ms,
                   no header). Subject-to-group mapping via CRUCES_META_CSV.
  NAGOYA_RRI_DIR : folder with *_RRi.txt files (space-separated: rel_time_s rri_s).
  NAGOYA_META_CSV: metadata with columns Subject_ID, Group, Start_Time (HH:MM:SS).

Output
------
  data/benchmarks/loco_cv_results.csv  — columns: Excluded, Acc, AUC
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV      = os.path.join(BASE, 'data', 'benchmarks', 'loco_cv_results.csv')

# Raw RRi data — adjust if your local paths differ
CETRAM_RRI_DIR  = "/Users/leo/HRV-Complexity/CETRAM/results_strict_analysis/cleaned_detections"
CRUCES_RRI_DIR  = "/Users/leo/HRV-Complexity/Cruces/public_release/data/processed/RRi"
CRUCES_META_CSV = "/Users/leo/HRV-Complexity/Multicenter/public_release/data/spain_metrics.csv"
NAGOYA_RRI_DIR  = "/Users/leo/HRV-Complexity/Nagoya/public_release/data/processed_rri"
NAGOYA_META_CSV = "/Users/leo/HRV-Complexity/Nagoya/public_release/data/metadata/metadata.csv"

# ── Hyperparameters ───────────────────────────────────────────────────────────

SEQ_LEN    = 400    # beats per window
STRIDE     = 50     # sliding window step (beats) — same for all centers
NAGOYA_WIN = (16, 20)  # clock hours: 16:00–20:00
RRI_MIN    = 0.3    # seconds
RRI_MAX    = 2.0    # seconds
CENTER_RRI = 0.9    # subtract to zero-centre input
EPOCHS     = 40
BATCH_SIZE = 128
LR         = 0.001

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# ── Model ─────────────────────────────────────────────────────────────────────

class ResNet1D(nn.Module):
    """
    Lightweight 1D ResNet for binary classification of RRi sequences.
      Stem  : Conv1d(1→32, k=7) + ReLU
      Res1  : identity shortcut, 32→32 channels
      Res2  : stride-2 shortcut, 32→64 channels (halves sequence length)
      Head  : GlobalAveragePool → FC(64→32→1)
    """
    def __init__(self):
        super().__init__()
        self.stem      = nn.Conv1d(1, 32, 7, padding=3)
        self.res1      = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1))
        self.res2      = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1))
        self.shortcut  = nn.Conv1d(32, 64, 1, stride=2)
        self.gap       = nn.AdaptiveAvgPool1d(1)
        self.head      = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = F.relu(x + self.res1(x))
        x = F.relu(self.shortcut(x) + self.res2(x))
        return self.head(self.gap(x).squeeze(-1))


# ── Data loading ──────────────────────────────────────────────────────────────

def make_windows(rri_s):
    """
    Apply bandpass filter, zero-centre, then slice into SEQ_LEN windows
    with STRIDE step.  Returns (N_windows × SEQ_LEN) array or None if too short.
    """
    rri = rri_s[(rri_s > RRI_MIN) & (rri_s < RRI_MAX)] - CENTER_RRI
    if len(rri) < SEQ_LEN:
        return None
    return np.array([rri[s:s + SEQ_LEN]
                     for s in range(0, len(rri) - SEQ_LEN + 1, STRIDE)])


def load_cetram():
    """
    CETRAM: *_cleaned.csv files organised into PD/ and Control/ subfolders.
    'sample' column contains ECG peak indices at 1000 Hz.
    RRi = diff(sample) / 1000  →  seconds.
    """
    print("Loading CETRAM...")
    data = {}
    for grp_label, label in [('PD', 1), ('Control', 0)]:
        for fpath in glob.glob(os.path.join(CETRAM_RRI_DIR, grp_label, '*.csv')):
            sid = os.path.basename(fpath).replace('_cleaned.csv', '')
            samples = pd.read_csv(fpath)['sample'].values
            rri     = np.diff(samples) / 1000.0
            wins    = make_windows(rri)
            if wins is not None:
                data[sid] = (wins, label)
    print(f"  {len(data)} subjects loaded")
    return data


def load_cruces():
    """
    Cruces: per-subject *.csv files (single column, no header, RRi in ms).
    Group mapping from last two columns (Subject, Group) of spain_metrics.csv.
    """
    print("Loading Cruces...")
    meta = pd.read_csv(CRUCES_META_CSV)
    grp_map = meta.set_index('Subject')['Group'].str.lower().map(
        {'pd': 1, 'control': 0, 'parkinson': 1, 'other': 0}).to_dict()
    data = {}
    for fpath in glob.glob(os.path.join(CRUCES_RRI_DIR, '*.csv')):
        sid = os.path.basename(fpath).replace('.csv', '')
        if sid not in grp_map:
            continue
        rri  = pd.read_csv(fpath, header=None)[0].values / 1000.0
        wins = make_windows(rri)
        if wins is not None:
            data[sid] = (wins, grp_map[sid])
    print(f"  {len(data)} subjects loaded")
    return data


def load_nagoya():
    """
    Nagoya: *_RRi.txt files (space-separated: rel_time_s  rri_s).
    Filtered to clock hours NAGOYA_WIN = (16, 20) using recording Start_Time
    from metadata.  This matches the 16–20h window used for handcrafted features.
    """
    print(f"Loading Nagoya ({NAGOYA_WIN[0]:02d}:00–{NAGOYA_WIN[1]:02d}:00)...")
    meta    = pd.read_csv(NAGOYA_META_CSV)
    meta_d  = meta.set_index('Subject_ID').to_dict('index')
    data    = {}
    for fpath in glob.glob(os.path.join(NAGOYA_RRI_DIR, '*_RRi.txt')):
        sid = os.path.basename(fpath).replace('_RRi.txt', '')
        if sid not in meta_d:
            continue
        info   = meta_d[sid]
        label  = 1 if str(info['Group']).upper() == 'PD' else 0
        try:
            st  = datetime.strptime(str(info['Start_Time']).strip(), '%H:%M:%S')
            off = st.hour * 3600 + st.minute * 60 + st.second
        except Exception:
            off = 9 * 3600  # fallback to 09:00

        df    = pd.read_csv(fpath, sep=r'\s+', header=None,
                            names=['rel_time', 'rri'])
        abs_h = ((df['rel_time'].values + off) % (24 * 3600)) / 3600.0
        mask  = (abs_h >= NAGOYA_WIN[0]) & (abs_h < NAGOYA_WIN[1])
        rri   = df.loc[mask, 'rri'].values
        wins  = make_windows(rri)
        if wins is not None:
            data[sid] = (wins, label)
    print(f"  {len(data)} subjects loaded")
    return data


# ── Training & evaluation ─────────────────────────────────────────────────────

def run_fold(fold_name, train_dicts, test_dict):
    print(f"\n--- FOLD: Leave-{fold_name}-Out ---")

    # Pool all training windows
    X_tr, y_tr = [], []
    for d in train_dicts:
        for sid, (wins, lab) in d.items():
            X_tr.extend(wins)
            y_tr.extend([lab] * len(wins))
    X_tr = np.array(X_tr, dtype=np.float32)
    y_tr = np.array(y_tr, dtype=np.float32)

    counts  = np.bincount(y_tr.astype(int))
    weights = np.array([1.0 / counts[int(t)] for t in y_tr])
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(y_tr))

    X_t = torch.tensor(X_tr).unsqueeze(1).to(DEVICE)
    y_t = torch.tensor(y_tr).unsqueeze(1).to(DEVICE)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=BATCH_SIZE, sampler=sampler)

    model = ResNet1D().to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.BCEWithLogitsLoss()

    print(f"  Training on {len(y_tr)} windows "
          f"(PD={counts[1]}, Control={counts[0]})...")
    for epoch in range(EPOCHS):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS}")

    # Subject-level evaluation via mean-probability voting
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for sid, (wins, lab) in test_dict.items():
            bx    = torch.tensor(wins, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            probs = torch.sigmoid(model(bx)).cpu().numpy().flatten()
            vote  = float(np.mean(probs))
            y_true.append(lab)
            y_pred.append(1 if vote > 0.5 else 0)
            y_score.append(vote)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    print(f"  Results — Acc={acc:.3f}  AUC={auc:.3f}")
    return {'Excluded': fold_name, 'Acc': acc, 'AUC': auc}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  1D-ResNet LOCO Cross-Validation")
    print(f"  SEQ_LEN={SEQ_LEN}  STRIDE={STRIDE}  EPOCHS={EPOCHS}")
    print(f"  Nagoya window: {NAGOYA_WIN[0]:02d}:00–{NAGOYA_WIN[1]:02d}:00")
    print("=" * 60)

    sites = {
        'Chile': load_cetram(),
        'Spain': load_cruces(),
        'Japan': load_nagoya(),
    }
    print(f"\nSubjects per center: "
          + "  ".join(f"{k}={len(v)}" for k, v in sites.items()))

    results = []
    results.append(run_fold('Chile',
                            [sites['Spain'], sites['Japan']], sites['Chile']))
    results.append(run_fold('Spain',
                            [sites['Chile'], sites['Japan']], sites['Spain']))
    results.append(run_fold('Japan',
                            [sites['Chile'], sites['Spain']], sites['Japan']))

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")


if __name__ == '__main__':
    main()
