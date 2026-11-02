      # Data Provenance

      Last synced: 2026-11-01 09:57:34

      ## Source mapping

      | Multicenter file | Source |
      |---|---|
      | chile_mse.csv | CETRAM pipeline=sqi: CETRAM/public_release/results/entropy/sample/MSE_curves_sample.csv |
      | chile_metrics.csv | CETRAM pipeline=sqi: CETRAM/public_release/results/metrics/HRV_metrics_cleaned.csv |
      | chile_demographics.csv | CETRAM/public_release/data/metadata/subject_demographics.csv |
      | spain_mse.csv | Cruces/public_release/results/entropy/sample/MSE_curves_sample.csv |
      | spain_metrics.csv | Cruces/public_release/results/metrics/HRV_metrics.csv |
      | spain_demographics.csv | Cruces/public_release/data/metadata/subject_demographics.csv |
      | japan_metadata.csv | Nagoya/public_release/data/metadata/metadata.csv |
      | japan_window_mse.csv | Derived: Nagoya/calculations/japan_4h_mse_curves.csv (Window 18–22 h) |
      | japan_morning_mse.csv | Derived: Nagoya/calculations/japan_4h_mse_curves.csv (Window 07–11 h) |
      | japan_evolution.csv | Derived: japan_4h_nauc.csv + Full_HRV_Evolution_1min.csv |
      | japan_window_features.csv | Computed by compute_japan_window_features.py (reads Nagoya raw RRi) |
      | japan_recalc_metrics.csv | Computed by compute_japan_fullday_hrv.py (long-running, run manually when Nagoya RRi changes) |
      | deidentified_clinical_consolidated.xlsx | **Manual** — CETRAM clinical database |
      | sample_signals/ | **Manual** — anonymised CETRAM RRi traces |
      | benchmarks/ | **Manual** — DL LOCO benchmark results |

      ## Sync log

      ```
        ✓  OK     chile_mse.csv                             (unchanged)
✓  OK     chile_metrics.csv                         (unchanged)
✓  OK     chile_demographics.csv                    (unchanged)
→  COPY  spain_mse.csv                             [CHANGED]  Cruces/public_release/results/entropy/sample/MSE_curves_sample.csv
→  COPY  spain_metrics.csv                         [CHANGED]  Cruces/public_release/results/metrics/HRV_metrics.csv
✓  OK     spain_demographics.csv                    (unchanged)
⚠  SKIP   japan_metadata.csv                        (source is OLDER than dest — use --force to override)
→  DERIVE  japan_window_mse.csv                      (47 subjects, Window 18–22 h)
→  DERIVE  japan_morning_mse.csv                     (39 subjects, Window 07–11 h)
→  DERIVE  japan_evolution.csv                       (50 subjects, 995 window-rows)
→  COMPUTE japan_window_features.csv                 (47 rows written)
✓  OK     japan_recalc_metrics.csv                    (up to date)
ℹ  MANUAL deidentified_clinical_consolidated.xlsx   (CETRAM clinical DB — update from source when clinical data changes)
ℹ  MANUAL sample_signals/                           (hand-picked CETRAM RRi traces for Figure 2 — curated, not derived)
ℹ  MANUAL benchmarks/                               (DL LOCO results — run benchmark_dl_loco.py when raw data changes)
      ```
