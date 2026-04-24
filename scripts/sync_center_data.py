"""
sync_center_data.py
===================
Syncs pre-computed metrics from each center's public_release into
Multicenter/public_release/data/ and writes a PROVENANCE.md record.

Run this script whenever any center pipeline is updated, then re-run
the Multicenter figures.

Usage
-----
    cd Multicenter/public_release/scripts
    python sync_center_data.py                  # dry-run (shows what would change)
    python sync_center_data.py --apply          # actually copy
    python sync_center_data.py --apply --force  # copy even if source is older

File origins
------------
CETRAM  (direct copies):
    results/entropy/sample/MSE_curves_sample.csv → chile_mse.csv
    results/metrics/HRV_metrics_cleaned.csv       → chile_metrics.csv
    data/metadata/subject_demographics.csv        → chile_demographics.csv

Cruces  (direct copies):
    results/entropy/sample/MSE_curves_sample.csv → spain_mse.csv
    results/metrics/HRV_metrics.csv              → spain_metrics.csv
    data/metadata/subject_demographics.csv       → spain_demographics.csv

Nagoya  (direct copies):
    data/metadata/metadata.csv                   → japan_metadata.csv
    calculations/japan_4h_nauc.csv       ┐
    results/metrics/Full_HRV_Evolution_1min.csv  ┘ → japan_evolution.csv (merged)
    calculations/japan_4h_mse_curves.csv         → japan_afternoon_mse.csv (16–20 h)
    calculations/japan_4h_mse_curves.csv         → japan_morning_mse.csv  (07–11 h)

Computed inside Multicenter (NOT synced, but re-run automatically):
    compute_japan_window_features.py             → japan_afternoon_features.csv

Manual / frozen (origin pre-dates this sync mechanism):
    japan_recalc_metrics.csv    — full-24h NeuroKit2 HRV metrics for Nagoya;
                                  update manually from Nagoya's per-subject
                                  calculation if you reprocess Nagoya data.
    deidentified_clinical_consolidated.xlsx — CETRAM clinical data; update
                                  manually when clinical database changes.
    sample_signals/             — anonymised CETRAM RRi traces for Figure 2;
                                  update manually.
    benchmarks/                 — DL LOCO results; update manually.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR   = Path(__file__).resolve().parent
ROOT       = THIS_DIR.parent                          # Multicenter/public_release
DATA_DIR   = ROOT / "data"
HRV_ROOT   = ROOT.parent.parent                       # HRV-Complexity/

CETRAM     = HRV_ROOT / "CETRAM"  / "public_release"
CRUCES     = HRV_ROOT / "Cruces"  / "public_release"
NAGOYA     = HRV_ROOT / "Nagoya"  / "public_release"

# ── CETRAM pipeline selection ──────────────────────────────────────────────────
# "sqi"  — original pipeline (SQI template-match, Pan-Tompkins peaks)
# "bsqi" — Eduardo Berríos consensus criterion (Ho et al. 2025) — under evaluation
CETRAM_PIPELINE = "bsqi"

# ---------------------------------------------------------------------------
# Copy rules — built at import time from CETRAM_PIPELINE
# ---------------------------------------------------------------------------

_CETRAM_SOURCES = {
    "sqi": {
        "mse":     CETRAM / "results/entropy/sample/MSE_curves_sample.csv",
        "metrics": CETRAM / "results/metrics/HRV_metrics_cleaned.csv",
    },
    "bsqi": {
        "mse":     CETRAM / "results/entropy_bsqi/sample/MSE_curves_sample.csv",
        "metrics": CETRAM / "results/metrics/HRV_metrics_bsqi.csv",
    },
}

DIRECT_COPIES = [
    # CETRAM (pipeline-dependent)
    (_CETRAM_SOURCES[CETRAM_PIPELINE]["mse"],     "chile_mse.csv"),
    (_CETRAM_SOURCES[CETRAM_PIPELINE]["metrics"], "chile_metrics.csv"),
    (CETRAM / "data/metadata/subject_demographics.csv", "chile_demographics.csv"),
    # Cruces
    (CRUCES / "results/entropy/sample/MSE_curves_sample.csv", "spain_mse.csv"),
    (CRUCES / "results/metrics/HRV_metrics.csv",              "spain_metrics.csv"),
    (CRUCES / "data/metadata/subject_demographics.csv",       "spain_demographics.csv"),
    # Nagoya (direct)
    (NAGOYA / "data/metadata/metadata.csv",                   "japan_metadata.csv"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def file_status(src: Path, dst: Path) -> str:
    """Return a one-word status: MISSING_SRC / NEW / CHANGED / OLDER / OK."""
    if not src.exists():
        return "MISSING_SRC"
    if not dst.exists():
        return "NEW"
    if md5(src) != md5(dst):
        src_mtime = src.stat().st_mtime
        dst_mtime = dst.stat().st_mtime
        return "CHANGED" if src_mtime >= dst_mtime else "OLDER"
    return "OK"


def copy_file(src: Path, dst: Path, dry_run: bool, force: bool) -> str:
    status = file_status(src, dst)
    if status == "MISSING_SRC":
        return f"  ✗  SKIP   {dst.name:40s}  (source not found: {src})"
    if status == "OK":
        return f"  ✓  OK     {dst.name:40s}  (unchanged)"
    if status == "OLDER" and not force:
        return f"  ⚠  SKIP   {dst.name:40s}  (source is OLDER than dest — use --force to override)"
    action = "COPY" if not dry_run else "WOULD COPY"
    if not dry_run:
        shutil.copy2(src, dst)
    return f"  →  {action}  {dst.name:40s}  [{status}]  {src.relative_to(HRV_ROOT)}"


# ---------------------------------------------------------------------------
# Derived Japan files
# ---------------------------------------------------------------------------

def derive_japan_afternoon_mse(dry_run: bool) -> str:
    """Filter japan_4h_mse_curves.csv to Window_start_h == 16 (16–20 h window)."""
    src   = NAGOYA / "calculations/japan_4h_mse_curves.csv"
    dst   = DATA_DIR / "japan_afternoon_mse.csv"
    label = "japan_afternoon_mse.csv"
    if not src.exists():
        return f"  ✗  SKIP   {label:40s}  (source not found: {src})"
    df = pd.read_csv(src)
    df["Group"] = df["Group"].str.lower().map({"control": "Control", "pd": "PD"}).fillna(df["Group"])  # normalise capitalisation
    df_aft = (df[df["Window_start_h"] == 16]
              .rename(columns={"Scale": "Scales"})
              [["Subject", "Group", "Scales", "MSE"]]
              .sort_values(["Subject", "Scales"])
              .reset_index(drop=True))
    if not dry_run:
        df_aft.to_csv(dst, index=False)
    n = df_aft.Subject.nunique()
    action = "DERIVE" if not dry_run else "WOULD DERIVE"
    return f"  →  {action}  {label:40s}  ({n} subjects, Window 16–20 h)"


def derive_japan_morning_mse(dry_run: bool) -> str:
    """Filter japan_4h_mse_curves.csv to Window_start_h == 7 (07–11 h window)."""
    src   = NAGOYA / "calculations/japan_4h_mse_curves.csv"
    dst   = DATA_DIR / "japan_morning_mse.csv"
    label = "japan_morning_mse.csv"
    if not src.exists():
        return f"  ✗  SKIP   {label:40s}  (source not found: {src})"
    df = pd.read_csv(src)
    df["Group"] = df["Group"].str.lower().map({"control": "Control", "pd": "PD"}).fillna(df["Group"])  # normalise capitalisation
    df_morn = (df[df["Window_start_h"] == 7]
               .rename(columns={"Scale": "Scales"})
               [["Subject", "Group", "Scales", "MSE"]]
               .sort_values(["Subject", "Scales"])
               .reset_index(drop=True))
    if not dry_run:
        df_morn.to_csv(dst, index=False)
    n = df_morn.Subject.nunique()
    action = "DERIVE" if not dry_run else "WOULD DERIVE"
    return f"  →  {action}  {label:40s}  ({n} subjects, Window 07–11 h)"


def derive_japan_evolution(dry_run: bool) -> str:
    """
    Merge japan_4h_nauc.csv with per-window HR/SDNN/RMSSD from
    Full_HRV_Evolution_1min.csv to produce japan_evolution.csv.

    The evolution file aggregates 1-min stats into 4-h window summaries
    (median per window) and joins with the nAUC complexity values.
    """
    src_nauc = NAGOYA / "calculations/japan_4h_nauc.csv"
    src_evo  = NAGOYA / "results/metrics/Full_HRV_Evolution_1min.csv"
    dst      = DATA_DIR / "japan_evolution.csv"
    label    = "japan_evolution.csv"

    for s in (src_nauc, src_evo):
        if not s.exists():
            return f"  ✗  SKIP   {label:40s}  (source not found: {s})"

    nauc = pd.read_csv(src_nauc)
    evo  = pd.read_csv(src_evo)
    _norm = lambda s: s.str.lower().map({"control": "Control", "pd": "PD"}).fillna(s)
    nauc["Group"] = _norm(nauc["Group"])
    evo["Group"]  = _norm(evo["Group"])

    # HR is derived directly from n_beats: each nauc window is exactly 4 hours
    # (minimum 4000 beats required by compute_4h_windows_nauc.jl), so:
    #   HR [bpm] = n_beats / 240 min
    nauc["HR"] = nauc["n_beats"] / 240.0

    # SDNN and RMSSD: aggregate 1-min values over the 4-hour window
    # [W, W+4). nauc windows slide at 1-hour steps (0,1,...,23).
    records = []
    for _, row in nauc.iterrows():
        w_start = row["Window_start_h"]
        sub_evo = evo[(evo["Subject"] == row["Subject"]) &
                      (evo["Time_h"] >= w_start) &
                      (evo["Time_h"] <  w_start + 4)]
        records.append({
            "Subject":         row["Subject"],
            "Window_start_h":  w_start,
            "SDNN":            sub_evo["SDNN"].median() if len(sub_evo) else float("nan"),
            "RMSSD":           sub_evo["RMSSD"].median() if len(sub_evo) else float("nan"),
        })
    agg = pd.DataFrame(records)

    merged = nauc.merge(agg, on=["Subject", "Window_start_h"], how="left")
    merged = merged[["Subject", "Group", "Window_start_h",
                     "n_beats", "nAUC_1_20", "HR", "SDNN", "RMSSD"]]

    # Verify against existing file if it exists
    if dst.exists() and not dry_run:
        existing = pd.read_csv(dst)
        if set(merged.columns) != set(existing.columns):
            return (f"  ⚠  WARN   {label:40s}  column mismatch with existing file — "
                    f"inspect before applying")

    if not dry_run:
        merged.to_csv(dst, index=False)
    n = merged.Subject.nunique()
    action = "DERIVE" if not dry_run else "WOULD DERIVE"
    return f"  →  {action}  {label:40s}  ({n} subjects, {len(merged)} window-rows)"


def recompute_afternoon_features(dry_run: bool) -> str:
    """Re-run compute_japan_window_features.py to update japan_afternoon_features.csv."""
    script = THIS_DIR / "compute_japan_window_features.py"
    dst    = DATA_DIR / "japan_afternoon_features.csv"
    label  = "japan_afternoon_features.csv"
    if not script.exists():
        return f"  ✗  SKIP   {label:40s}  (script not found: {script})"
    if dry_run:
        return f"  →  WOULD RUN  {label:40s}  (compute_japan_window_features.py)"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, cwd=str(THIS_DIR)
    )
    if result.returncode != 0:
        return (f"  ✗  FAIL   {label:40s}  "
                f"(compute_japan_window_features.py exited {result.returncode})\n"
                f"     stderr: {result.stderr.strip()[:200]}")
    n = len(pd.read_csv(dst)) if dst.exists() else "?"
    return f"  →  COMPUTE {label:40s}  ({n} rows written)"


# ---------------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------------

def write_provenance(log_lines: list[str], dry_run: bool) -> None:
    if dry_run:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = textwrap.dedent(f"""\
        # Data Provenance

        Last synced: {now}

        ## Source mapping

        | Multicenter file | Source |
        |---|---|
        | chile_mse.csv | CETRAM pipeline={CETRAM_PIPELINE}: {_CETRAM_SOURCES[CETRAM_PIPELINE]["mse"].relative_to(HRV_ROOT)} |
        | chile_metrics.csv | CETRAM pipeline={CETRAM_PIPELINE}: {_CETRAM_SOURCES[CETRAM_PIPELINE]["metrics"].relative_to(HRV_ROOT)} |
        | chile_demographics.csv | CETRAM/public_release/data/metadata/subject_demographics.csv |
        | spain_mse.csv | Cruces/public_release/results/entropy/sample/MSE_curves_sample.csv |
        | spain_metrics.csv | Cruces/public_release/results/metrics/HRV_metrics.csv |
        | spain_demographics.csv | Cruces/public_release/data/metadata/subject_demographics.csv |
        | japan_metadata.csv | Nagoya/public_release/data/metadata/metadata.csv |
        | japan_afternoon_mse.csv | Derived: Nagoya/calculations/japan_4h_mse_curves.csv (Window 16–20 h) |
        | japan_morning_mse.csv | Derived: Nagoya/calculations/japan_4h_mse_curves.csv (Window 07–11 h) |
        | japan_evolution.csv | Derived: japan_4h_nauc.csv + Full_HRV_Evolution_1min.csv |
        | japan_afternoon_features.csv | Computed by compute_japan_window_features.py (reads Nagoya raw RRi) |
        | japan_recalc_metrics.csv | Computed by compute_japan_fullday_hrv.py (long-running, run manually when Nagoya RRi changes) |
        | deidentified_clinical_consolidated.xlsx | **Manual** — CETRAM clinical database |
        | sample_signals/ | **Manual** — anonymised CETRAM RRi traces |
        | benchmarks/ | **Manual** — DL LOCO benchmark results |

        ## Sync log

        ```
        {chr(10).join(log_lines)}
        ```
    """)
    (DATA_DIR / "PROVENANCE.md").write_text(content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sync center data into Multicenter/data/")
    parser.add_argument("--apply", action="store_true",
                        help="Actually copy/derive files (default: dry-run)")
    parser.add_argument("--force", action="store_true",
                        help="Copy even when source is older than destination")
    args = parser.parse_args()
    dry_run = not args.apply

    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  Multicenter Data Sync")
    if dry_run:
        print("  MODE: DRY RUN — pass --apply to make changes")
    else:
        print("  MODE: APPLY")
    print(f"  CETRAM pipeline: {CETRAM_PIPELINE.upper()}"
          + ("  ← canonical" if CETRAM_PIPELINE == "sqi" else "  ← UNDER EVALUATION"))
    print(SEP)

    log = []

    # ── Direct copies ──────────────────────────────────────────────────────
    print("\n── Direct copies ─────────────────────────────────────────────────")
    for src, dst_name in DIRECT_COPIES:
        line = copy_file(src, DATA_DIR / dst_name, dry_run, args.force)
        print(line)
        log.append(line)

    # ── Derived Japan files ────────────────────────────────────────────────
    print("\n── Derived Japan files ───────────────────────────────────────────")
    for fn in (derive_japan_afternoon_mse, derive_japan_morning_mse, derive_japan_evolution):
        line = fn(dry_run)
        print(line)
        log.append(line)

    # ── Computed within Multicenter ────────────────────────────────────────
    print("\n── Computed within Multicenter ───────────────────────────────────")
    line = recompute_afternoon_features(dry_run)
    print(line)
    log.append(line)

    # ── Long-running computed (stale check only, never run automatically) ────
    print("\n── Long-running computed files ───────────────────────────────────")
    recalc_dst = DATA_DIR / "japan_recalc_metrics.csv"
    nagoya_rri_dir = NAGOYA / "data/processed_rri"
    if not recalc_dst.exists():
        line = ("  ✗  MISSING japan_recalc_metrics.csv                    "
                "  → run: python compute_japan_fullday_hrv.py")
    elif nagoya_rri_dir.exists():
        rri_files = list(nagoya_rri_dir.glob("*_RRi.txt"))
        newest_rri = max(f.stat().st_mtime for f in rri_files) if rri_files else 0
        if newest_rri > recalc_dst.stat().st_mtime:
            line = ("  ⚠  STALE  japan_recalc_metrics.csv                    "
                    "  → Nagoya RRi files are newer; run compute_japan_fullday_hrv.py")
        else:
            line = "  ✓  OK     japan_recalc_metrics.csv                    (up to date)"
    else:
        line = ("  ℹ  MANUAL japan_recalc_metrics.csv                    "
                "  (Nagoya RRi dir not accessible from this machine)")
    print(line)
    log.append(line)

    # ── Truly manual files (non-derived, update from source only) ────────
    print("\n── Manual files (not auto-synced) ────────────────────────────────")
    manual = [
        ("deidentified_clinical_consolidated.xlsx",
         "CETRAM clinical DB — update from source when clinical data changes"),
        ("sample_signals/",
         "hand-picked CETRAM RRi traces for Figure 2 — curated, not derived"),
        ("benchmarks/",
         "DL LOCO results — run benchmark_dl_loco.py when raw data changes"),
    ]
    for fname, note in manual:
        line = f"  ℹ  MANUAL {fname:40s}  ({note})"
        print(line)
        log.append(line)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    if dry_run:
        print("  Dry run complete. Re-run with --apply to execute changes.")
    else:
        write_provenance(log, dry_run=False)
        print(f"  Sync complete. Provenance written → {DATA_DIR / 'PROVENANCE.md'}")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
