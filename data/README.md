# data/

Derived, de-identified tables. Participants appear only under study codes. Groups are
`Control` and `PD` (Cruces also lists six records labelled `Other`, excluded from every
analysis). Cohort prefixes in file names follow the country: `chile_` = CETRAM,
`spain_` = Cruces, `japan_` = Nagoya.

"Made by" names the script that writes the file: scripts in `scripts/cohorts/` run from
the raw recordings (`python reproduce.py cohorts`), the `compute_*` scripts from the RR
series (`python reproduce.py derived`). Files marked *given* were extracted from clinical
records, which are not distributed.

## Per-participant tables

| File | Contents | Made by |
|---|---|---|
| `chile_mse.csv`, `spain_mse.csv` | rcMSE curve, scales 1–20 (`MSE`, `Scales`) | `julia/run_rcmse.jl` (via `reproduce.py cohorts`) |
| `japan_window_mse.csv` | rcMSE curve in the 18:00–22:00 window (primary), scales 1–20 | `cohorts/nagoya_windows.py` |
| `japan_afternoon_mse.csv`, `japan_morning_mse.csv` | the same in 16:00–20:00 and 07:00–11:00 | `cohorts/nagoya_windows.py` |
| `chile_metrics.csv`, `spain_metrics.csv` | HRV indices (NeuroKit2 names, `HRV_*`) | `cohorts/cetram_hrv.py`, `cohorts/cruces_hrv.py` |
| `japan_window_features.csv` | HRV indices, age, sex, HR in the 18:00–22:00 window | `compute_japan_window_features.py` |
| `chile_demographics.csv`, `spain_demographics.csv` | age and sex (CETRAM: age on the day of the ECG) | given |
| `japan_metadata.csv` | age, sex, recording start time, file name | given |
| `japan_evolution.csv` | complexity (nAUC 1–20) and median HR, SDNN, RMSSD in 4-h windows sliding by 1 h | `cohorts/nagoya_windows.py` |
| `japan_15min_windows.csv` | complexity and HRV in consecutive 15-min windows over 24 h | `compute_nagoya_windows.py` |
| `japan_scale_profile.csv` | rcMSE to scale 60 in the 18:00–22:00 window | `compute_nagoya_windows.py` |
| `japan_overlap_grid.csv` | complexity in overlapping 15-min windows on a 45-s grid | `compute_overlap_grid.py` |
| `sdnn_min_all_centers.csv` | minimum and percentiles of SDNN and CV over 100/200/300-beat windows | `compute_sdnn_min.py` |
| `length_ladder.csv` | index on series truncated to 250–4000 beats, with split halves | `compute_length_ladder.py` |
| `split_half_reliability.csv` | index on the first and second half of each series, each with its own tolerance | `compute_split_half.py` |
| `r_sweep.csv` | index over tolerance k (or fixed r in ms) and embedding dimension m, with split halves | `compute_r_sweep.py` |
| `spectral_analysis.csv` | spectral exponent, and rcMSE of the series against matched 1/f noise | `compute_spectral_analysis.py` |
| `irregular_rhythm_flags.csv` | share of beat-to-beat variance at 0.35–0.50 cycles/beat, and flag | `compute_irregular_rhythm_flags.py` |
| `cetram_detector_curves.csv` | rcMSE (scales 1–5) with Brammer and Pan–Tompkins beat marks | `provenance/pt_entropy.py`, from a re-run of the CETRAM beat detection with each detector |

## Summary tables

| File | Contents | Made by |
|---|---|---|
| `fig7_window_ladder.csv`, `fig7_dist_summaries.csv`, `fig7_hourly_auc.csv`, `fig7_window_scan.csv`, `fig7_cetram_headtohead.csv` | Fig. 7 panels | `compute_fig7_inputs.py` |
| `fig7_overlap.csv` | Fig. 7, overlapping windows | `build_overlap_summary.py` |
| `spectral_psd_curves.csv` | median spectra per cohort and reference noise (Suppl. Fig. S2) | `compute_spectral_analysis.py` |
| `cruces_iniguez_replication.csv`, `cruces_iniguez_medians.csv` | source-study HRV metrics in Cruces: AUC, 95% CI, p, medians | `compute_cruces_replication.py` |
| `natural_earth/ne_110m_land.geojson` | land outlines for the Fig. 1 map (Natural Earth, public domain) | — |

## Conventions

- RR intervals are in ms; intervals outside 300–2000 ms are not analysed.
- Complexity index: nAUC = trapz(curve, τ = 1…n) / n, scales 1–5 in CETRAM and Cruces,
  1–20 in Nagoya, with r = 0.2 SD of the whole series held fixed across scales.
- AUC is oriented so that values above 0.5 mean lower values in PD.
