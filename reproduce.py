#!/usr/bin/env python3
"""Reproduce the figures and the numbers of the manuscript.

    python reproduce.py            # figures from the shipped tables in data/, then the claim check
    python reproduce.py check      # only the claim check (every number quoted in the text)
    python reproduce.py derived    # rebuild data/ from the original series (needs HRV_DATA_ROOT)
    python reproduce.py import     # re-import the per-cohort result tables (needs HRV_DATA_ROOT)
    python reproduce.py all        # import + derived + figures + check

Figures go to figures/<manuscript figure>/. A figure whose inputs are not distributed
(raw ECG, clinical table) is reported as skipped; see DATA_AVAILABILITY.md.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import paths  # noqa: E402

# (script, what it makes, inputs that are not distributed)
FIGURES = [
    ("fig1_map.py", "Fig. 1  elements: world map", []),
    ("fig1_icons.py", "Fig. 1  elements: modality icons", []),
    ("fig1_ecg_traces.py", "Fig. 1  elements: illustrative ECG strips", []),
    ("fig1_elements.py", "Fig. 1  elements: data-driven panels", [paths.CETRAM_PEAKS]),
    ("fig1_layout.py", "Fig. 1  layout mock-up (final figure assembled in Affinity Designer)", []),
    ("fig2_archetypes.py", "Fig. 2  signal archetypes", []),
    ("fig3_curves_index.py", "Fig. 3  rcMSE curves and complexity index", []),
    ("fig4_age.py", "Fig. 4  complexity and age", []),
    ("fig5_parameters.py", "Fig. 5  entropy-parameter sensitivity", []),
    ("fig6_detector.py", "Fig. 6  beat detector", [paths.CETRAM_RAW_ECG]),
    ("fig7_recording_length.py", "Fig. 7  recording length", []),
    ("fig8_step1_hrv_table.py", "Fig. 8  step 1/5  HRV table", []),
    ("fig8_step2_scale_correlations.py", "Fig. 8  step 2/5  per-scale correlations", []),
    ("fig8_step3_correlations.py", "Fig. 8  step 3/5  complexity vs HRV correlations", []),
    ("fig8_step4_incremental_value.py", "Fig. 8  step 4/5  incremental value", []),
    ("fig8_step5_cross_cohort.py", "Fig. 8  step 5/5  cross-cohort consistency", []),
    ("fig8_hrv_relationships.py", "Fig. 8  complexity and conventional HRV", []),
    ("fig9_clinical.py", "Fig. 9  clinical correlations", [paths.CLINICAL_XLSX]),
    ("figS1_hr_normalization.py", "Fig. S1 heart-rate normalization", []),
    ("figS2_spectral.py", "Fig. S2 spectral character", []),
    ("figS3_cruces_context.py", "Fig. S3 Cruces and the source study", []),
    ("figS4_length.py", "Fig. S4 recording length", []),
    ("figS5_ppg_scales.py", "Fig. S5 per-scale AUC by modality", []),
    ("figS6_S7_cross_cohort.py", "Fig. S6-S7 cross-cohort transfer", []),
    ("figS8_validation.py", "Fig. S8 methods validation", []),
    ("figS9_traces.py", "Fig. S9 raw-trace review", [paths.CETRAM_RAW_ECG]),
]

# Rebuild data/ from the original beat/RR series, in dependency order.
DERIVED = [
    ("compute_japan_window_features.py", "Nagoya HRV, 18:00-22:00  -> japan_window_features.csv"),
    ("compute_nagoya_windows.py", "Nagoya 15-min windows  -> japan_15min_windows.csv, japan_scale_profile.csv"),
    ("compute_sdnn_min.py", "minimum SDNN  -> sdnn_min_all_centers.csv"),
    ("compute_length_ladder.py", "length ladder  -> length_ladder.csv"),
    ("compute_overlap_grid.py", "Nagoya overlapping windows  -> japan_overlap_grid.csv"),
    ("build_overlap_summary.py", "  -> fig7_overlap.csv"),
    ("compute_fig7_inputs.py", "Fig. 7 inputs  -> fig7_*.csv"),
    ("compute_spectral_analysis.py", "spectra  -> spectral_analysis.csv, spectral_psd_curves.csv"),
    ("compute_irregular_rhythm_flags.py", "irregular rhythm  -> irregular_rhythm_flags.csv"),
    ("compute_split_half.py", "split-half reliability  -> split_half_reliability.csv"),
    ("compute_cruces_replication.py", "Cruces source-study metrics  -> cruces_iniguez_replication.csv"),
    ("compute_r_sweep.py", "entropy-parameter sweep  -> r_sweep.csv  (slow: about 30 min)"),
]
DERIVED_NEEDS = [paths.CETRAM_PEAKS, paths.CRUCES_RRI, paths.NAGOYA_RRI, paths.NAGOYA_META]


# Fixed timestamps in SVG/PDF output and a fixed hash seed, so reruns are byte-identical
# (matplotlibrc in this folder fixes the SVG element ids).
ENV = dict(os.environ, SOURCE_DATE_EPOCH="0", PYTHONHASHSEED="0")


def run(script, label):
    t = time.time()
    print(f"\n>>> {label}   [scripts/{script}]", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", script)], cwd=HERE, env=ENV)
    if r.returncode:
        sys.exit(f"\n!!! scripts/{script} failed (exit {r.returncode})")
    print(f"    done in {time.time() - t:.0f} s", flush=True)


def figures():
    skipped = []
    for script, label, needs in FIGURES:
        missing = [p for p in needs if not paths.have(p)]
        if missing:
            skipped.append((label, missing))
            print(f"\n--- skipped {label}: needs {', '.join(missing)}")
            continue
        run(script, label)
    return skipped


def derived():
    missing = [p for p in DERIVED_NEEDS if not paths.have(p)]
    if missing:
        sys.exit("\n'derived' needs the original series, not found at:\n  " + "\n  ".join(missing) +
                 "\nSet HRV_DATA_ROOT (see DATA_AVAILABILITY.md).")
    for script, label in DERIVED:
        run(script, label)


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "figures"
    if step not in ("figures", "check", "derived", "import", "all"):
        sys.exit(__doc__)
    print(f"data/           {paths.DATA}\nHRV_DATA_ROOT   {paths.DATA_ROOT}")
    skipped = []
    if step in ("import", "all"):
        run("import_cohort_tables.py", "import the per-cohort result tables")
    if step in ("derived", "all"):
        derived()
    if step in ("figures", "all"):
        skipped = figures()
    if step in ("figures", "check", "all"):
        run("verify_manuscript_claims.py", "check every number quoted in the manuscript")
    if skipped:
        print("\nSkipped (inputs not distributed, see DATA_AVAILABILITY.md):")
        for label, _ in skipped:
            print("  ", label)
    print("\nFinished.")


if __name__ == "__main__":
    main()
