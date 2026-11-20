# Figure 1 elements

Pieces for the hand-drawn overview (`../Fig1_overview.af`). Data-driven panels by
`scripts/fig1_elements.py` (needs the original CETRAM beat series); text is live and editable,
backgrounds are transparent. Colours match the other figures (control #4C72B0, PD #C44E52).

| File | Use in the layout | Source |
|---|---|---|
| rr_input.svg | "RR-interval series" input, left | CETRAM control C028 (closest to the control median index), 150 artefact-free beats |
| A_variability.svg | branch A, conventional variability | same participant, whole record: RR histogram, mean, +-1 SD |
| B_coarse_graining.svg | branch B, coarse-graining | same 150 beats averaged in non-overlapping blocks of tau = 1, 3, 5 |
| B_entropy_curve.svg | branch B, entropy vs scale | same participant, rcMSE scales 1-5, r = 0.2 SD fixed; shaded area = complexity index |
| result_cetram.svg | outcome panel, one cohort | group-mean rcMSE +- SE, CETRAM 42 C / 27 PD |
| result_three_cohorts.svg | outcome panel, all cohorts | as Fig. 3A, B, D (Nagoya 18:00-22:00, scales 1-20) |
| rr_control.svg, rr_pd.svg | optional example traces (replace the stylised heart traces) | C028 and P011, closest to their group median index, same y-axis |

Counts for the cohort labels: Santiago 42 C / 27 PD (15-min ECG); Bilbao 21 C / 31 PD
(7.40-min finger PPG during fMRI); Nagoya 21 disease controls / 26 PD (24-h chest strap,
18:00-22:00 window).

## Map
| File | Use | Source |
|---|---|---|
| world_map_sites.svg | map with pins, leader lines and editable cohort labels | Natural Earth 1:110m land (public domain), Equal Earth projection, Antarctica omitted; `scripts/fig1_map.py` |
| world_map_pins.svg | same map, pins only (place your own labels) | as above |

Pins use the cohort colours of scripts/figstyle.py: Santiago amber #E69F00, Bilbao olive #7F7F00, Nagoya cyan #17BECF, all chosen away from the Control (#4C72B0) and PD (#C44E52) colours and checked for colour-vision deficiency; land is neutral grey (#CFCFCF).
Natural Earth needs no attribution; "Made with Natural Earth" is customary if you want to credit it.

## Modality icons
Original filled pictograms drawn for this figure (no third-party artwork; free to use and modify).
`scripts/fig1_icons.py`. Sensors are filled with the cohort colour; `*_neutral.svg`
versions use light grey instead.

| File | Cohort |
|---|---|
| icon_ecg_rest.svg | Santiago: bust with two chest electrodes and an ECG trace (resting ECG) |
| icon_ppg_fmri.svg | Bilbao: hand with the index finger in a pulse sensor showing a pulse wave, small MRI badge (finger PPG during fMRI) |
| icon_chest_strap_24h.svg | Nagoya: bust with chest strap and sensor pod, clock badge (24-h recording) |

## Illustrative ECG strips
`ecg_trace_control.svg` (blue), `ecg_trace_pd.svg` (red) and `ecg_traces_pair.svg`:
synthetic strips with identical beats and different beat timing (irregular, multiscale
in the control strip; nearly metronomic in the PD strip), exaggerated for legibility.
Not patient data. `scripts/fig1_ecg_traces.py`.

## Layout mock-up
`../figure1_layout.svg` arranges these elements as a starting point for the final figure;
`scripts/fig1_layout.py`.
