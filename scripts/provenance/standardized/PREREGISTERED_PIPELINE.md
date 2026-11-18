# Standardized interbeat-interval processing — fixed before any group comparison
Agreed with Leo 2026-11-01 09:33 (America/Santiago). Written before Cruces group results were computed
under this pipeline. Diagnostics used to design it (scratch/ppg_fiducials/) read no group labels.
Disclosure: on 2026-10-31 an exploratory Cruces group comparison was run (raw / 20%-rule cleaning /
median-of-3; scratch/ppg_test/), before this pipeline was designed. Its results are not used to choose any step here.

1. Beat times
   - CETRAM: R peaks, Brammer detector (NeuroKit2 default), unchanged.
   - Cruces: PPG maximum-first-derivative point on the upstroke of each Elgendi-located pulse,
     parabolic sub-sample refinement, nk.ppg_clean(500 Hz) signal (Peralta 2019; Peng 2015).
   - Nagoya: device RR intervals (POLAR), unchanged; recording gaps split the series into segments.
2. Ectopic/misdetection correction: Lipponen & Tarvainen (2019), NeuroKit2 signal_fixpeaks(method="Kubios",
   iterative=True), applied within contiguous segments. Already applied in CETRAM; now applied to Cruces and Nagoya.
3. Physiological guard: 300-2000 ms (deletion).
4. Irregular-rhythm exclusion (label-blind, all cohorts, on the analysed segment after steps 2-3):
   exclude if >= 0.30 of the detrended interval variance (31-beat moving-mean removed; Welch) lies at
   0.35-0.50 cycles/beat, i.e. the beat-to-beat spectrum is as flat as white noise (~0.30). Rationale:
   HRV presupposes sinus rhythm (Task Force 1996); PPG/strap RR cannot adjudicate rhythm.
5. Complexity: rcMSE, m = 2, r = 0.2 SD of the original series fixed across scales, nAUC over scales
   1-5 (CETRAM, Cruces) and 1-20 (Nagoya 18:00-22:00), unchanged.
6. Sensitivity only (not primary): Chang & Yu adaptive median-of-3 smoother on Cruces intervals.
