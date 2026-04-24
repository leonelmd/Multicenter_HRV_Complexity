      # Data Provenance

      Last synced: 2026-04-24 15:18:53

      ## Source mapping

      | Multicenter file | Source |
      |---|---|
      | chile_mse.csv | CETRAM pipeline=bsqi: CETRAM/public_release/results/entropy_bsqi/sample/MSE_curves_sample.csv |
      | chile_metrics.csv | CETRAM pipeline=bsqi: CETRAM/public_release/results/metrics/HRV_metrics_bsqi.csv |
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
        ✓  OK     chile_mse.csv                             (unchanged)
✓  OK     chile_metrics.csv                         (unchanged)
✓  OK     chile_demographics.csv                    (unchanged)
✓  OK     spain_mse.csv                             (unchanged)
✓  OK     spain_metrics.csv                         (unchanged)
✓  OK     spain_demographics.csv                    (unchanged)
✓  OK     japan_metadata.csv                        (unchanged)
→  DERIVE  japan_afternoon_mse.csv                   (45 subjects, Window 16–20 h)
→  DERIVE  japan_morning_mse.csv                     (39 subjects, Window 07–11 h)
→  DERIVE  japan_evolution.csv                       (50 subjects, 995 window-rows)
→  COMPUTE japan_afternoon_features.csv              (46 rows written)
✓  OK     japan_recalc_metrics.csv                    (up to date)
ℹ  MANUAL deidentified_clinical_consolidated.xlsx   (CETRAM clinical DB — update from source when clinical data changes)
ℹ  MANUAL sample_signals/                           (hand-picked CETRAM RRi traces for Figure 2 — curated, not derived)
ℹ  MANUAL benchmarks/                               (DL LOCO results — run benchmark_dl_loco.py when raw data changes)
      ```
