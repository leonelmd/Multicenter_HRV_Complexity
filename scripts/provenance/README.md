# Provenance scripts

These scripts produced numbers that are quoted in the manuscript but were written in a
time-limited sandbox under `scratch/` (which is not tracked). They are released here
verbatim so that every reported number has its code in the repository. Several of them
expect the sandbox directory layout (`ppg/`, `fid/`, `dur/`, `std/`) or absolute
sandbox paths and do not run unmodified from this repository.

| Folder | Manuscript content |
|---|---|
| `standardized/` | Pre-registered interbeat-interval pipeline (PREREGISTERED_PIPELINE.md), irregular-rhythm screen. The flag itself is reproduced by `scripts/compute_irregular_rhythm_flags.py`, which runs from the released data. |
| `cruces_diagnosis/` | Cruces scale-structure diagnostics (Results, "The photoplethysmographic cohort"): falling-curve fractions, surrogates, respiratory-band share, sampling and fiducial checks. |
| `ppg_fiducials/` | Label-blind choice of the PPG fiducial point and ectopic correction for Cruces. |
| `adjudication_2026-11-17/` | CETRAM trace-review exclusions: `cetram_manuscript_numbers.py` recomputes every CETRAM number in the text (run from `public_release/`). |
| `pt_entropy.py` | Pan-Tompkins versus Brammer comparison (Results, "Sensitivity to the beat-detection stage"). |
