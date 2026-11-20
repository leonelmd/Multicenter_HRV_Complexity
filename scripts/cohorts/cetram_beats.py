#!/usr/bin/env python3
"""CETRAM: raw ECG -> cleaned beat times.

For each recording in $HRV_DATA_ROOT/cetram/raw/{Control,PD}/:
  1. NeuroKit2 ecg_clean; R peaks with the "neurokit" detector (R apex).
  2. Recording-level quality gate: two-detector agreement bSQI >= 0.80
     (neurokit vs pantompkins1985, 50-ms tolerance; Ho et al. 2025).
  3. Beat-level correction of ectopic, missed and extra beats (Lipponen & Tarvainen 2019,
     NeuroKit2 signal_fixpeaks, method "Kubios").
  4. Physiological guard 300-2000 ms: offending intervals replaced by linear interpolation.
  5. At least 650 clean beats and 80 % clean intervals; recordings excluded after trace
     review are listed in cetram_exclusions.csv.
Writes $HRV_DATA_ROOT/cetram/beats/{Control,PD}/<ID>_cleaned.csv (column `sample`, 1 kHz)
and beats/cleaning_summary.csv.

    python scripts/cohorts/cetram_beats.py [--peak-method neurokit]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import neurokit2 as nk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CETRAM_RAW_ECG, CETRAM_PEAKS, require  # noqa: E402

DATA_DIRS = [{"path": os.path.join(CETRAM_RAW_ECG, g), "group": g} for g in ("Control", "PD")]
OUTPUT_DIR = CETRAM_PEAKS
FS = 1000
MIN_CLEAN_BEATS = 650
RRI_MIN_MS, RRI_MAX_MS = 300, 2000

# R-peak detector used for the ANALYSIS series (overridable with --peak-method).
#   - any NeuroKit2 ecg_peaks method name, e.g. "neurokit", "pantompkins1985", "kalidas2017"
#   - "ecgdetectors:<name>" for py-ecg-detectors (Porr et al.), e.g. "ecgdetectors:two_average"
# NOTE: NeuroKit2 "pantompkins1985" returns the peak of the moving-window integral, not the
# R apex; on CETRAM ~37 % of its marks fall on the S trough (~35 ms late). See
# data/detector_robustness/README.md. "neurokit" returns the R apex of ecg_clean.
PEAK_METHOD = "neurokit"

# bSQI parameters (Ho et al. 2025) — two-detector consensus criterion.
# The bSQI detector pair is FIXED and independent of PEAK_METHOD, so that changing the
# analysis detector never changes which recordings pass the quality gate.
BSQI_METHODS      = ("neurokit", "pantompkins1985")   # (detector A, detector B)
BSQI_THRESHOLD    = 0.80   # minimum fraction of A-peaks confirmed by B
BSQI_TOLERANCE_MS = 50     # ms window within which two detectors must agree on a peak

# Recording-level exclusions decided by visual review of the raw traces. Applied after the
# bSQI gate, with the same FAIL status. Criteria: non-sinus rhythm (frequent ectopy) or
# minutes without any ECG signal; per-subject detail in the CSV.
ADJUDICATED_EXCLUSIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cetram_exclusions.csv")


def load_adjudicated_exclusions(path=ADJUDICATED_EXCLUSIONS):
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    return dict(zip(d["Subject"], d["Criterion"]))


def detect_peaks(ecg_raw, ecg_clean, method, fs=FS):
    """R-peak detection with the method named in PEAK_METHOD.

    NeuroKit2 methods run on nk.ecg_clean (as in the original pipeline).
    py-ecg-detectors methods ("ecgdetectors:<name>") run on the raw ECG, as that package
    expects unfiltered input (and as benchmarked in Porr & Macfarlane 2024 and
    Weinstein, Rodino & Otero 2026). Their fiducial is detector-specific: check it.
    """
    if method.startswith("ecgdetectors:"):
        from ecgdetectors import Detectors          # pip install py-ecg-detectors
        name = method.split(":", 1)[1]
        func = getattr(Detectors(fs), name if name.endswith("_detector") else name + "_detector")
        return np.asarray(func(np.asarray(ecg_raw, float)), dtype=int)
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method=method)
    return np.asarray(info["ECG_R_Peaks"], dtype=int)


def compute_bsqi(ecg_raw, ecg_clean, fs=FS, tol_ms=BSQI_TOLERANCE_MS, methods=BSQI_METHODS):
    """
    Beat Signal Quality Index (bSQI) — Ho et al. (2025).

    Two-detector consensus between methods[0] (A) and methods[1] (B) on the same
    nk.ecg_clean-preprocessed signal. For each peak in A, checks whether B has a peak
    within ±tol_ms. bSQI is the fraction of A-peaks confirmed by B.

    Returns (bsqi_score, {method: peaks}) so the caller can reuse a detector's peaks
    when PEAK_METHOD is one of the bSQI pair.
    """
    peaks = {m: detect_peaks(ecg_raw, ecg_clean, m, fs) for m in methods}
    peaks_A, peaks_B = peaks[methods[0]], peaks[methods[1]]

    if len(peaks_A) < 50 or len(peaks_B) < 50:
        return np.nan, peaks

    tol = int(tol_ms / 1000.0 * fs)
    matches = sum(1 for p in peaks_A if np.any(np.abs(peaks_B - p) <= tol))
    return matches / len(peaks_A), peaks


def process_subject_robust(file_path, subject_id, group):
    try:
        data = pd.read_csv(file_path, delimiter='\t', skiprows=3, header=None)
        ecg_signal = data.iloc[:, 5].to_numpy()

        ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=FS)

        # --- bSQI recording-level quality gate (Ho et al. 2025) ---
        # Rejects recordings where two independent detectors disagree on > 20% of
        # peaks, indicating global signal quality insufficient for HRV analysis.
        bsqi_score, bsqi_peaks = compute_bsqi(ecg_signal, ecg_clean, FS, BSQI_TOLERANCE_MS)

        # --- Analysis R-peaks: PEAK_METHOD (reused from bSQI when it is one of the pair) ---
        if PEAK_METHOD in bsqi_peaks:
            r_peaks = bsqi_peaks[PEAK_METHOD]
        else:
            r_peaks = detect_peaks(ecg_signal, ecg_clean, PEAK_METHOD, FS)

        if np.isnan(bsqi_score) or bsqi_score < BSQI_THRESHOLD:
            return {
                "Subject":      subject_id,
                "Group":        group,
                "Peak_Method":  PEAK_METHOD,
                "bSQI":         round(float(bsqi_score) if not np.isnan(bsqi_score) else -1, 3),
                "Total_Beats":  len(r_peaks),
                "Kubios_Fixes": 0,
                "Clean_Beats":  0,
                "Clean_Ratio":  0.0,
                "Status":       "FAIL",
                "Reason":       "bSQI %.3f < %.2f" % (bsqi_score if not np.isnan(bsqi_score) else -1, BSQI_THRESHOLD),
                "R_Peaks":      [],
            }

        # --- Beat-level artifact correction: Lipponen & Tarvainen (2019) ---
        # Corrects ectopic beats, missed beats, extra beats, and long-short pairs.
        # This replaces the Malik local-median filter, which removed beats and
        # created spurious merged intervals in the saved peak sequence.
        try:
            artifacts_info, peaks_clean = nk.signal_fixpeaks(
                r_peaks, sampling_rate=FS, iterative=True, method="Kubios", show=False
            )
            artifact_keys = ("ectopic", "missed", "extra", "longshort")
            n_artifacts = sum(len(artifacts_info.get(k, [])) for k in artifact_keys)
        except Exception as e:
            print(f"  Warning: Kubios correction failed for {subject_id} ({e}). Using raw peaks.")
            peaks_clean = r_peaks
            n_artifacts = 0

        # --- Quality assessment on corrected peaks ---
        rri = np.diff(peaks_clean) / FS * 1000
        # Physiological safety filter — Kubios handles most cases; this is a hard guard.
        valid_physio = (rri >= RRI_MIN_MS) & (rri <= RRI_MAX_MS)
        n_clean_beats = int(np.sum(valid_physio))
        clean_ratio = n_clean_beats / len(rri) if len(rri) > 0 else 0

        # --- Physiological guard on the saved series ------------------------------
        # Offending intervals are replaced by linear interpolation, never deleted:
        # deleting a beat and re-differencing would merge two intervals. clean_ratio is
        # computed above from the unmodified series, so this cannot change PASS/FAIL.
        n_guarded = int(np.sum(~valid_physio))
        if n_guarded > 0 and valid_physio.sum() >= 10:
            rri_fixed = rri.astype(float).copy()
            bad = ~valid_physio
            rri_fixed[bad] = np.interp(
                np.flatnonzero(bad), np.flatnonzero(valid_physio), rri[valid_physio]
            )
            # Rebuild the peak sequence so diff(sample) returns the corrected RRi.
            steps = np.rint(rri_fixed * FS / 1000.0).astype(np.int64)
            peaks_clean = np.concatenate(
                ([int(peaks_clean[0])], int(peaks_clean[0]) + np.cumsum(steps))
            )

        # Recording-level quality is pre-screened by bSQI above.
        # These checks catch pathological edge cases (extremely short recording,
        # residual impossible intervals after Kubios).
        status = "PASS"
        reason = ""
        if n_clean_beats < MIN_CLEAN_BEATS:
            status = "FAIL"
            reason = f"Too few clean beats ({n_clean_beats}) after Kubios correction"
        elif clean_ratio < 0.80:
            status = "FAIL"
            reason = f"Outlier ratio {100*(1-clean_ratio):.1f}% after Kubios correction"

        return {
            "Subject":      subject_id,
            "Group":        group,
            "Peak_Method":  PEAK_METHOD,
            "bSQI":         round(bsqi_score, 3),
            "Total_Beats":  len(r_peaks),
            "Kubios_Fixes": n_artifacts,
            "Physio_Guarded": n_guarded,
            "Clean_Beats":  n_clean_beats,
            "Clean_Ratio":  round(clean_ratio, 4),
            "Status":       status,
            "Reason":       reason,
            # Full corrected peak sequence (no beat removal): rcMSE uses diff(sample).
            "R_Peaks":      peaks_clean if status == "PASS" else [],
        }
    except Exception as e:
        return {"Subject": subject_id, "Group": group, "Status": "ERROR", "Reason": str(e)}

# --- Main Execution ---
def main():
    global PEAK_METHOD, OUTPUT_DIR
    ap = argparse.ArgumentParser(description="CETRAM ECG cleaning: bSQI gate + R-peaks + Kubios correction")
    ap.add_argument("--peak-method", default=PEAK_METHOD,
                    help='NeuroKit2 ecg_peaks method (e.g. "neurokit") or "ecgdetectors:<name>" '
                         f'(default: {PEAK_METHOD})')
    ap.add_argument("--output-dir", default=OUTPUT_DIR,
                    help="where to write the beat files (default: $HRV_DATA_ROOT/cetram/beats)")
    args = ap.parse_args()
    PEAK_METHOD, OUTPUT_DIR = args.peak_method, os.path.abspath(args.output_dir)
    print(f"R-peak method: {PEAK_METHOD} | bSQI pair: {BSQI_METHODS} | output: {OUTPUT_DIR}")
    require(CETRAM_RAW_ECG, "the raw CETRAM ECG")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    excluded = load_adjudicated_exclusions()
    print(f"Adjudicated exclusions: {sorted(excluded)}")

    for source in DATA_DIRS:
        base_path = source["path"]
        group = source["group"]
        if not os.path.exists(base_path): continue
            
        for item in sorted(os.listdir(base_path)):
            if item == "All": continue
            item_path = os.path.join(base_path, item)
            
            txt_file = None
            sub_id = item
            if os.path.isfile(item_path) and item.endswith(".txt"):
                txt_file = item_path
                sub_id = item.replace("_raw.txt", "").replace(".txt", "")
            elif os.path.isdir(item_path):
                txt_file = next((os.path.join(item_path, f) for f in os.listdir(item_path) 
                                 if f.startswith("opensignals_") and f.endswith(".txt")), None)
            
            if txt_file:
                print(f"Processing {sub_id}...")
                res = process_subject_robust(txt_file, sub_id, group)
                if res.get("Status") == "PASS" and sub_id in excluded:
                    res["Status"] = "FAIL"
                    res["Reason"] = f"trace review: {excluded[sub_id]}"
                results.append(res)
                
                if res["Status"] == "PASS":
                    save_dir = os.path.join(OUTPUT_DIR, group)
                    os.makedirs(save_dir, exist_ok=True)
                    pd.DataFrame({"sample": res["R_Peaks"], "tipo": "cleaned"}).to_csv(
                        os.path.join(save_dir, f"{sub_id}_cleaned.csv"), index=False
                    )

    df_report = pd.DataFrame(results).drop(columns=["R_Peaks"], errors="ignore")
    df_report.to_csv(os.path.join(OUTPUT_DIR, "cleaning_summary.csv"), index=False)
    print("\nData Cleaning Complete.")
    print(df_report["Status"].value_counts())

if __name__ == "__main__":
    main()
