#!/usr/bin/env python3
"""Import the per-cohort result tables into data/.

Each cohort has its own processing repository (raw signal -> beat series -> rcMSE
curves and HRV indices; see README, "Where the shipped tables come from"). This script
copies their outputs into data/ under the names used here, derives the Nagoya window
tables, and records the source and SHA-256 of every input in data/PROVENANCE.md.

Needs the per-cohort repositories under HRV_DATA_ROOT (paths.py).

    python scripts/import_cohort_tables.py
"""
import hashlib
import os
import shutil
from datetime import datetime

import pandas as pd

from paths import DATA, DATA_ROOT, require

CETRAM = os.path.join(DATA_ROOT, "CETRAM", "public_release")
CRUCES = os.path.join(DATA_ROOT, "Cruces", "public_release")
NAGOYA = os.path.join(DATA_ROOT, "Nagoya", "public_release")

COPIES = [  # (source, file in data/)
    (f"{CETRAM}/results/entropy/sample/MSE_curves_sample.csv", "chile_mse.csv"),
    (f"{CETRAM}/results/metrics/HRV_metrics_cleaned.csv", "chile_metrics.csv"),
    (f"{CETRAM}/data/metadata/subject_demographics.csv", "chile_demographics.csv"),
    (f"{CRUCES}/results/entropy/sample/MSE_curves_sample.csv", "spain_mse.csv"),
    (f"{CRUCES}/results/metrics/HRV_metrics.csv", "spain_metrics.csv"),
    (f"{CRUCES}/data/metadata/subject_demographics.csv", "spain_demographics.csv"),
    (f"{NAGOYA}/data/metadata/metadata.csv", "japan_metadata.csv"),
]
NAGOYA_CURVES = f"{NAGOYA}/calculations/japan_4h_mse_curves.csv"   # rcMSE, 4-h windows
NAGOYA_NAUC = f"{NAGOYA}/calculations/japan_4h_nauc.csv"
NAGOYA_EVOLUTION = f"{NAGOYA}/results/metrics/Full_HRV_Evolution_1min.csv"
WINDOWS = {18: "japan_window_mse.csv",      # 18:00-22:00, pre-specified primary window
           16: "japan_afternoon_mse.csv",   # 16:00-20:00 alternative
           7: "japan_morning_mse.csv"}      # 07:00-11:00

CANON = {"pd": "PD", "control": "Control", "controls": "Control", "hc": "Control", "other": "Other"}


def canon_groups(df):
    if "Group" in df.columns:
        g = df["Group"].astype(str).str.strip()
        df["Group"] = g.str.lower().map(CANON).fillna(g)
    return df


def rel(path):
    return os.path.relpath(path, DATA_ROOT)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    log = []
    for src, name in COPIES:
        require(src, f"the per-cohort table for {name}")
        dst = os.path.join(DATA, name)
        shutil.copyfile(src, dst)
        canon_groups(pd.read_csv(dst)).to_csv(dst, index=False)   # Nagoya writes 'control'
        log.append((name, rel(src), sha256(src)))

    curves = canon_groups(pd.read_csv(require(NAGOYA_CURVES, "the Nagoya 4-h rcMSE curves")))
    for start, name in WINDOWS.items():
        (curves[curves.Window_start_h == start].rename(columns={"Scale": "Scales"})
         [["Subject", "Group", "Scales", "MSE"]].sort_values(["Subject", "Scales"])
         .to_csv(os.path.join(DATA, name), index=False))
        log.append((name, f"{rel(NAGOYA_CURVES)} (window starting {start}:00)", sha256(NAGOYA_CURVES)))

    # japan_evolution.csv: complexity per 4-h window (sliding by 1 h) with the median of the
    # 1-min HR, SDNN and RMSSD inside the window. HR comes from the interval series, not
    # from a beat count, because window coverage is incomplete in some recordings.
    nauc = canon_groups(pd.read_csv(require(NAGOYA_NAUC, "the Nagoya 4-h index table")))
    evo = canon_groups(pd.read_csv(require(NAGOYA_EVOLUTION, "the Nagoya 1-min HRV table")))
    rows = []
    for r in nauc.itertuples():
        w = evo[(evo.Subject == r.Subject) & (evo.Time_h >= r.Window_start_h) & (evo.Time_h < r.Window_start_h + 4)]
        rows.append(dict(Subject=r.Subject, Window_start_h=r.Window_start_h,
                         HR=w.HR.median(), SDNN=w.SDNN.median(), RMSSD=w.RMSSD.median()))
    (nauc.merge(pd.DataFrame(rows), on=["Subject", "Window_start_h"], how="left")
     [["Subject", "Group", "Window_start_h", "n_beats", "nAUC_1_20", "HR", "SDNN", "RMSSD"]]
     .to_csv(os.path.join(DATA, "japan_evolution.csv"), index=False))
    log.append(("japan_evolution.csv", f"{rel(NAGOYA_NAUC)} + {rel(NAGOYA_EVOLUTION)}",
                sha256(NAGOYA_NAUC) + " / " + sha256(NAGOYA_EVOLUTION)))

    with open(os.path.join(DATA, "PROVENANCE.md"), "w") as f:
        f.write(f"# Provenance of the imported cohort tables\n\nImported {datetime.now():%Y-%m-%d %H:%M} "
                f"by scripts/import_cohort_tables.py.\nSources are relative to HRV_DATA_ROOT; "
                f"SHA-256 truncated to 16 hex digits.\n\n| data/ file | source | sha256 |\n|---|---|---|\n")
        for name, src, h in log:
            f.write(f"| {name} | {src} | {h} |\n")
    for name, _, _ in log:
        print("  imported", name)


if __name__ == "__main__":
    main()
