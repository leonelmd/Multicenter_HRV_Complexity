"""Text edits for the 2026-11-17 adjudicated exclusion (CETRAM n = 42 C / 27 PD).
Every replacement must match exactly once; the script aborts otherwise.
Numbers: numbers_post.csv (cetram_manuscript_numbers.py) and the figure-script logs."""
import sys, os
MS = sys.argv[1]
E = []
def e(f, old, new):
    E.append((f, old, new))

# ---------------------------------------------------------------- abstract
e("main.tex", "(ECG; 43 controls, 30 PD, 15 min)", "(ECG; 42 controls, 27 PD, 15 min)")
e("main.tex", "in the ECG cohort (AUC 0.681, $p=0.009$)", "in the ECG cohort (AUC 0.655, $p=0.031$)")
e("main.tex", "statistic fell to 0.469,\nbelow chance, whereas complexity retained 0.681.",
  "statistic fell to 0.480,\nbelow chance, whereas complexity retained 0.655.")

# ---------------------------------------------------------------- methods
M = "sections/02_methods.tex"
e(M, "together with its robustness to age, to outlying\ninterval series and to the entropy parameters.",
  "together with its robustness to age, to the recordings\nexcluded after raw-trace review and to the entropy parameters.")
e(M, r"Chile/CETRAM & 43 & 30 & $63.4\pm9.4$ & $64.1\pm11.4$", r"Chile/CETRAM & 42 & 27 & $63.1\pm9.4$ & $64.8\pm10.8$")
e(M, """contiguous. Forty-three controls and 30 PD participants had complete data, and because
the recording is short, scales 1--5 were used.""",
"""contiguous. Every recording whose analysed series had an SDNN above 100~ms, i.e., roughly
twice the short-term value expected at rest \\cite{TaskForce1996HRV}, was then reviewed on
the raw ECG, plotted in full at 30~s per row and in 10-s strips at 25~mm/s and 10~mm/mV, by
a clinician co-author (P.C.-C.) who was not blinded to group. Of the seven recordings
reviewed, four were excluded, on two criteria: frequent ectopic beats, with approximately
18\\% of intervals premature (one PD, one control), such that the series no longer reflects
sinus rhythm \\cite{TaskForce1996HRV}, and minutes without any recognizable QRS complex, in
which the detector had marked baseline noise (two PD). The other three were retained,
because the rhythm was sinus in each (Section~\\ref{sec:outliers}). The review was applied
after the full-sample results were known, and those results are therefore reported
alongside. Forty-two controls and 27 PD participants were analysed, and because the
recording is short, scales 1--5 were used.""")
e(M, "Clinical information was available for 28 of the 30 PD participants",
  "Clinical information was available for 25 of the 27 PD participants")
e(M, """levodopa-equivalent daily dose (LEDD), the last computable for the 11
participants whose records give doses.""",
"""levodopa-equivalent daily dose (LEDD), the last computable for 21 participants from the
regimen recorded in their clinical files at the date of the ECG.""")
e(M, """excluding them gives AUC 0.670 ($p=0.017$) in CETRAM, 0.653 ($p=0.070$) in Cruces and
0.751 ($p=0.006$) in Nagoya, against 0.681, 0.657 and 0.773 with all recordings.""",
"""excluding them gives AUC 0.641 ($p=0.057$) in CETRAM, 0.653 ($p=0.070$) in Cruces and
0.751 ($p=0.006$) in Nagoya, against 0.655, 0.657 and 0.773 with all recordings.""")
e(M, r"CETRAM ($\sim$1105 beats)", r"CETRAM ($\sim$1112 beats)")
e(M, "measured slopes are $-0.40$, $-0.43$ and $-1.03$ in CETRAM",
  "measured slopes are $-0.68$, $-0.43$ and $-1.03$ in CETRAM")
e(M, "the correlation with HR changes from $\\rho=-0.13$, $-0.29$ and $-0.27$ to $-0.48$,",
  "the correlation with HR changes from $\\rho=-0.22$, $-0.29$ and $-0.27$ to $-0.58$,")
e(M, "discriminative power of HR itself (AUC 0.583, 0.639 and 0.745)",
  "discriminative power of HR itself (AUC 0.615, 0.639 and 0.745)")

# ---------------------------------------------------------------- results
R = "sections/03_results.tex"
e(R, "The analysis included 73 CETRAM participants, namely, 43 controls and 30 PD,",
  "The analysis included 69 CETRAM participants, namely, 42 controls and 27 PD,")
e(R, """SDNN was higher in CETRAM PD
participants (median 33.5 vs 48.7 ms, AUC 0.339, $p=0.020$), i.e., in the direction""",
"""SDNN tended to be higher in CETRAM PD
participants (median 33.4 vs 47.9 ms, AUC 0.361, $p=0.053$), i.e., in the direction""")
e(R, """but not in CETRAM, where RMSSD was likewise
higher in PD (AUC 0.357, $p=0.040$). These inverted CETRAM dispersion contrasts are
examined in Section~\\ref{sec:outliers}; they persist after the recordings with outlying
interval statistics are excluded.""",
"""but not in CETRAM, where RMSSD likewise
tended to be higher in PD (AUC 0.381, $p=0.098$). These inverted CETRAM dispersion
contrasts, which reach nominal significance in the full sample of 73, are examined in
Section~\\ref{sec:outliers}.""")
e(R, "PD in CETRAM (median 1.170 vs 0.926; AUC 0.681, 95\\% CI 0.547--0.809, $p=0.009$)",
  "PD in CETRAM (median 1.171 vs 0.948; AUC 0.655, 95\\% CI 0.510--0.787, $p=0.031$)")
e(R, "preserved in CETRAM ($p=0.005$, adjusted AUC 0.672)", "preserved in CETRAM ($p=0.031$, adjusted AUC 0.646)")
e(R, "CETRAM: $p=0.009$, HR-adjusted $p=0.005$, HR-adjusted AUC 0.672.",
  "CETRAM: $p=0.031$, HR-adjusted $p=0.031$, HR-adjusted AUC 0.646.")
e(R, "namely, CETRAM controls ($\\rho=-0.23$, $p=0.14$) and PD ($\\rho=-0.01$,\n$p=0.96$)",
  "namely, CETRAM controls ($\\rho=-0.20$, $p=0.21$) and PD ($\\rho=-0.16$,\n$p=0.49$)")
old_out = R and open(os.path.join(MS, R)).read()
a = old_out.index("\\subsubsection{Outlying interval series in CETRAM}")
b = old_out.index("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{figS_params.png}")
e(R, old_out[a:b], """\\subsubsection{Recordings excluded after trace review}
\\label{sec:outliers}
Seven CETRAM recordings, four of them PD, had interval series with SDNN above 100~ms,
i.e., roughly twice the short-term value reported for healthy adults
\\cite{TaskForce1996HRV}, and the raw-trace review described in Methods resolved them as
follows. Two, one PD and one control, carried frequent ectopic beats, with approximately
18\\% of intervals premature, mostly ventricular in the PD recording and mostly
supraventricular in the control, so that their dispersion reflected a non-sinus rhythm
rather than heart-rate variability. Two PD recordings began with stretches of about 2 and
5~min without any recognizable QRS complex, in which the detector had marked baseline
noise; once the ECG was present, both showed a regular sinus rhythm. These four were
excluded. Of the three retained, two, one of them PD, showed sinus rhythm with tremor or
movement artefact confined to part of the record, and one control showed a clean sinus
rhythm with slow drifts of heart rate.

The exclusion matters for the complexity index because $r=0.2\\sigma_x$: an inflated SD
widens the tolerance, template matching becomes easier, and entropy falls for a reason
unrelated to the dynamics of the rhythm. With the four recordings included, the CETRAM
result is AUC 0.681 ($p=0.009$, $n=73$), and excluding, in addition, the three retained
recordings with SDNN above 100~ms gives 0.659 ($p=0.031$, $n=66$), against 0.655
($p=0.031$) in the analysed sample. Minimum SDNN is below chance in each case (0.469, 0.466
and 0.480, respectively). The inverted SDNN and RMSSD contrasts do not arise from these
recordings: they are present with all 73 (AUC 0.339, $p=0.020$, and 0.357, $p=0.040$), in
the analysed sample (0.361, $p=0.053$, and 0.381, $p=0.098$) and with all seven removed
(0.342, $p=0.032$, and 0.373, $p=0.084$).

""")
e(R, "and $m=1$--$3$ the AUC moves by at most 0.050 in CETRAM", "and $m=1$--$3$ the AUC moves by at most 0.061 in CETRAM")
e(R, "estimator is undefined for 90.4\\% of Cruces and 52.1\\% of CETRAM participants, which is",
  "estimator is undefined for 90.4\\% of Cruces and 55.1\\% of CETRAM participants, which is")
e(R, "optimized on the outcome; the per-cohort optima at $m=2$ disagree in any case, at\n$k=0.15$, 0.25 and 0.25 respectively.",
  "optimized on the outcome; the per-cohort optima at $m=2$ disagree in any case, at\n$k=0.125$, 0.25 and 0.25 respectively.")
e(R, "since it correlates with the interval standard deviation at $\\rho=+0.55$ to $+0.93$ and\ndrives CETRAM below chance, i.e., it measures amplitude rather than complexity.",
  "since it correlates with the interval standard deviation at $\\rho=+0.59$ to $+0.93$ and\nleaves CETRAM at or near chance, i.e., it measures amplitude rather than complexity.")
e(R, "most 0.050 in CETRAM, 0.076 in Cruces and 0.027 in Nagoya, and split-half reliability was",
  "most 0.061 in CETRAM, 0.076 in Cruces and 0.027 in Nagoya, and split-half reliability was")
e(R, "undefined for part of a cohort, namely, for 90.4\\% of Cruces and 52.1\\% of CETRAM\nparticipants",
  "undefined for part of a cohort, namely, for 90.4\\% of Cruces and 55.1\\% of CETRAM\nparticipants")
e(R, "Further, the per-cohort optima at $m=2$ differ ($k=0.15$, $0.25$ and\n$0.25$, respectively)",
  "Further, the per-cohort optima at $m=2$ differ ($k=0.125$, $0.25$ and\n$0.25$, respectively)")
e(R, """The group contrast was preserved (AUC 0.684, $p=0.008$, against
0.681, $p=0.009$ with the Brammer detector), so the CETRAM result does not depend on the
detector.""",
"""The group contrast was preserved (AUC 0.687, $p=0.009$, against
0.655, $p=0.031$ with the Brammer detector), so the CETRAM result does not depend on the
detector.""")
e(R, """(controls 1.61 at $\\tau=1$ to 1.32 at $\\tau=5$; PD 1.29 to
1.08), whereas with the Brammer detector it rose (controls 1.22 to 1.49; PD 0.90 to
1.22)""",
"""(controls 1.62 at $\\tau=1$ to 1.33 at $\\tau=5$; PD 1.31 to
1.10), whereas with the Brammer detector it rose (controls 1.24 to 1.50; PD 0.97 to
1.29)""")
e(R, "curve while leaving the group contrast nearly intact agrees", "curve while preserving the group contrast agrees")
e(R, "while leaving CETRAM below\nchance at every such tolerance (AUC 0.42--0.49).",
  "while leaving CETRAM at or near\nchance at every such tolerance (AUC 0.42--0.51).")
e(R, "scales, yet only 16\\% of its curves fall.", "scales, yet only 17\\% of its curves fall.")
e(R, "whereas in CETRAM real and surrogate series agree (16\\% and\n16\\%).",
  "whereas in CETRAM real and surrogate series nearly agree (17\\% and\n14\\%).")
e(R, "larger share of variance in the respiratory band than CETRAM (0.41 vs 0.32 of the total)",
  "larger share of variance in the respiratory band than CETRAM (0.41 vs 0.31 of the total)")
e(R, "i.e., from 0.647 at 250 beats to 0.665 at 1000,", "i.e., from 0.620 at 250 beats to 0.667 at 1000,")
e(R, "in both short cohorts split-half reliability stayed at\nor below 0.40 over these lengths",
  "in both short cohorts split-half reliability stayed at\nor below 0.50 over these lengths")
e(R, "identical metric to CETRAM gives AUC 0.469, i.e., below chance, against 0.681 for\ncomplexity",
  "identical metric to CETRAM gives AUC 0.480, i.e., below chance, against 0.655 for\ncomplexity")
e(R, "the fall from 0.905 in\nNagoya to 0.469 in CETRAM reflects both", "the fall from 0.905 in\nNagoya to 0.480 in CETRAM reflects both")
e(R, """complexity, in Nagoya, in the full CETRAM cohort and in CETRAM after excluding the
seven outlying interval series of Section~\\ref{sec:outliers}.""",
"""complexity, in Nagoya, in the analysed CETRAM sample and in CETRAM after additionally
excluding the three retained recordings with SDNN above 100~ms (Section~\\ref{sec:outliers}).""")
e(R, """Twenty-eight of the 30 CETRAM PD participants had matched clinical information.""",
  """Twenty-five of the 27 analysed CETRAM PD participants had matched clinical information.""")
e(R, """either unadjusted (all $q>0.98$) or after adjustment for age (all $q>0.18$;
Fig.~\\ref{fig:clinical}). The strongest age-adjusted association was with disease
duration ($\\rho=-0.44$, uncorrected $p=0.020$, $q=0.18$), lower complexity accompanying
longer disease. LEDD could be computed for 11 participants only. These analyses are
underpowered""",
"""either unadjusted (all $q\\geq0.28$) or after adjustment for age (all $q\\geq0.13$;
Fig.~\\ref{fig:clinical}). The strongest age-adjusted association was with disease
duration ($\\rho=-0.48$, uncorrected $p=0.014$, $q=0.13$), lower complexity accompanying
longer disease. LEDD, computable for 21 participants, was not associated with complexity
($\\rho=-0.28$, $p=0.22$). These analyses are underpowered""")
e(R, """Orthostatic hypotension, recorded as clinical history, was available for 25 of the 30 PD
participants, 11 with and 14 without.""",
"""Orthostatic hypotension, recorded as clinical history, was available for 22 of the 27 PD
participants, 9 with and 13 without.""")
e(R, """hypotension (median 0.80 vs 1.20; AUC 0.779, $p=0.020$), and the difference persisted
after excluding the four PD recordings with outlying interval series (AUC 0.833,
$p=0.012$), after excluding one entry imputed during data consolidation (AUC 0.757,
$p=0.038$), and with age and heart rate in a rank-based regression ($p=0.015$). The
autonomic clinical index recorded in the same files, available for 16 participants, was
not associated with complexity ($\\rho=0.07$, $p=0.79$).""",
"""hypotension (median 0.80 vs 1.21; AUC 0.778, $p=0.033$), and the difference persisted
after excluding the retained PD recording with SDNN above 100~ms (AUC 0.833, $p=0.012$),
but was weaker with age and heart rate in a rank-based regression with HC3 standard
errors ($p=0.055$). The autonomic clinical index recorded in the same files, available
for 14 participants, was not associated with complexity ($\\rho=-0.12$, $p=0.69$).""")
e(R, "Exploratory associations between the complexity index and clinical measures in\nthe 28 CETRAM PD participants",
  "Exploratory associations between the complexity index and clinical measures in\nthe 25 CETRAM PD participants")
e(R, "dose (LEDD) in the 11 participants whose records give doses, unadjusted and adjusted for",
  "dose (LEDD) in the 21 participants with a recorded regimen, unadjusted and adjusted for")

# ---------------------------------------------------------------- discussion
DI = "sections/04_discussion.tex"
e(DI, "where complexity retained AUC 0.681.", "where complexity retained AUC 0.655.")
e(DI, """marks on the S wave. Replacing it with the Brammer detector left the CETRAM group contrast
nearly unchanged (AUC 0.684 versus 0.681) but turned a falling entropy curve into a rising""",
"""marks on the S wave. Replacing it with the Brammer detector preserved the CETRAM group
contrast (AUC 0.687 versus 0.655) but turned a falling entropy curve into a rising""")
a = open(os.path.join(MS, DI)).read()
s0 = a.index("Separately, seven CETRAM participants, four of whom are PD")
s1 = a.index("\\subsubsection{Why the photoplethysmographic cohort is weaker}")
e(DI, a[s0:s1], """Separately, seven CETRAM recordings had implausibly high dispersion for rest (SDNN above
100~ms), and we reviewed their raw traces rather than treating them only as a sensitivity
set (Section~\\ref{sec:outliers}). Two reflected frequent ectopy and two contained minutes
without any ECG signal; these four were excluded, and the other three, which showed sinus
rhythm with partial artefact or slow drifts of heart rate, were retained. This bears upon
the complexity index, since an inflated SD inflates the tolerance and thereby lowers
entropy for reasons unrelated to the dynamics of the rhythm. With the four included, the
CETRAM result is AUC 0.681 ($p=0.009$), and with all seven excluded 0.659 ($p=0.031$),
against 0.655 ($p=0.031$) in the analysed sample, so the conclusion does not depend on
the decision.

The review has limits that we state plainly. It was carried out by a clinician co-author
on single-lead tracings, it was not blinded to group, and it was applied after the
full-sample results were known; it was also confined to recordings with outlying
dispersion, so that occasional ectopic beats in the remainder were handled only by the
automated correction. Since HRV indices presuppose a regular underlying rhythm
\\cite{TaskForce1996HRV}, short-ECG studies should inspect at least every recording whose
dispersion is implausible for rest, as we do here, alongside a rhythm screen such as the
irregular-rhythm flag of Section~\\ref{sec:pipeline}. The clinical records that might bear
on rhythm remain incomplete, since the cardiac-history field is blank for 12 of the 27 PD
participants and controls are not represented in the workbook at all.

""")
e(DI, "0.49, against 0.87 in CETRAM.", "0.49, against 0.86 in CETRAM.")
e(DI, "tolerance raises Cruces to AUC 0.68--0.72 while dropping CETRAM below chance; we did not",
  "tolerance raises Cruces to AUC 0.68--0.72 while leaving CETRAM at or near chance; we did not")
e(DI, "such a tolerance correlates with SD at $\\rho\\geq0.55$ in every cohort",
  "such a tolerance correlates with SD at $\\rho\\geq0.59$ in every cohort")
e(DI, "although this exploratory comparison rests on 25 participants and on a recorded history",
  "although this exploratory comparison rests on 22 participants and on a recorded history")
e(DI, "in CETRAM the PD group showed\nhigher, not lower, SDNN and RMSSD;",
  "in CETRAM the PD group tended\nto show higher, not lower, SDNN and RMSSD;")
e(DI, """cohorts, a model on the complexity index and heart rate reached held-out AUCs of 0.690 in
CETRAM, 0.685 in Cruces and 0.835 in Nagoya, whereas SDNN and RMSSD with heart rate did
not transfer to CETRAM (AUC 0.495).""",
"""cohorts, a model on the complexity index and heart rate reached held-out AUCs of 0.676 in
CETRAM, 0.688 in Cruces and 0.852 in Nagoya, whereas SDNN and RMSSD with heart rate did
not transfer to CETRAM (AUC 0.504).""")
e(DI, """(Section~\\ref{sec:interpretation}). Second, seven CETRAM participants yielded outlying
interval series of undetermined origin; excluding them leaves the CETRAM result at AUC
0.659 ($p=0.031$).""",
"""(Section~\\ref{sec:interpretation}). Second, the CETRAM result rests on a sample from which
four recordings were excluded after an unblinded review of the raw traces; with them it is
AUC 0.681 ($p=0.009$), and without any recording of outlying dispersion 0.659 ($p=0.031$).""")

# ---------------------------------------------------------------- supplementary
S = "sections/06_supplementary.tex"
e(S, "($p=0.30$--$0.72$)", "($p=0.30$--$0.63$)")
e(S, """0.690 [0.559--0.808] in CETRAM, 0.685 [0.536--0.831] in Cruces and 0.835 [0.703--0.940]
in Nagoya, against 0.583, 0.639 and 0.745 for heart rate alone; the gain over heart rate
(0.046--0.107) has confidence intervals that include zero in each cohort. SDNN and RMSSD
with heart rate did not transfer to CETRAM (0.495).""",
"""0.676 [0.548--0.802] in CETRAM, 0.688 [0.535--0.831] in Cruces and 0.852 [0.729--0.949]
in Nagoya, against 0.615, 0.639 and 0.745 for heart rate alone; the gain over heart rate
(0.049--0.106) has confidence intervals that include zero in each cohort. SDNN and RMSSD
with heart rate did not transfer to CETRAM (0.504).""")
e(S, """(a) Complexity index with heart rate: AUC 0.676--0.839 in
every direction, with no advantage for models trained on the 24-h recordings (Nagoya to
CETRAM 0.676, CETRAM to Nagoya 0.835). (b) SDNN and RMSSD with heart rate, which transfers
poorly whenever CETRAM is the training or test cohort (0.449--0.566).""",
"""(a) Complexity index with heart rate: AUC 0.668--0.855 in
every direction, with no advantage for models trained on the 24-h recordings (Nagoya to
CETRAM 0.668, CETRAM to Nagoya 0.855). (b) SDNN and RMSSD with heart rate, which transfers
poorly whenever CETRAM is the training or test cohort (0.475--0.575).""")
e(S, "cohort it was trained on (AUC 0.583 in CETRAM, 0.639 in Cruces, 0.745 in Nagoya).}",
  "cohort it was trained on (AUC 0.615 in CETRAM, 0.639 in Cruces, 0.745 in Nagoya).}")

# ---------------------------------------------------------------- tables
T = "tables/traditional_benchmark.tex"
e(T, "Complexity nAUC & 0.681 (0.009)", "Complexity nAUC & 0.655 (0.031)")
e(T, "HR              & 0.583 (0.232)", "HR              & 0.615 (0.111)")
e(T, "SDNN            & 0.339 (0.020)", "SDNN            & 0.361 (0.053)")
e(T, "RMSSD           & 0.357 (0.040)", "RMSSD           & 0.381 (0.098)")
e(T, "pNN50           & 0.386 (0.099)", "pNN50           & 0.401 (0.168)")
e(T, """The inverted CETRAM
SDNN and RMSSD values persist after excluding the seven participants whose interval series
are outliers of undetermined origin (Section~\\ref{sec:outliers}).""",
"""The CETRAM SDNN and
RMSSD contrasts are inverted, at nominal significance in the full sample of 73 before
the trace-review exclusions (Section~\\ref{sec:outliers}).""")
e("tables/primary_results_revised.tex",
  "Chile/CETRAM & 43 & 30 & 1.170 [1.043--1.284] & 0.926 [0.617--1.203] & 0.009 & 0.681 [0.547--0.809]",
  "Chile/CETRAM & 42 & 27 & 1.171 [1.054--1.288] & 0.948 [0.704--1.213] & 0.031 & 0.655 [0.510--0.787]")
e("tables/endpoint_sensitivity_revised.tex",
  "Chile/CETRAM & 0.681 & 0.009 & 0.672 & 0.005 & $-0.687$ (0.005)",
  "Chile/CETRAM & 0.655 & 0.031 & 0.646 & 0.031 & $-0.537$ (0.031)")

# ---------------------------------------------------------------- apply
texts = {}
bad = []
for f, old, new in E:
    t = texts.setdefault(f, open(os.path.join(MS, f)).read())
    n = t.count(old)
    if n != 1:
        bad.append((f, n, old[:90].replace("\n", " ")))
        continue
    texts[f] = t.replace(old, new)
if bad:
    for b in bad:
        print("NO UNIQUE MATCH:", b)
    sys.exit(1)
for f, t in texts.items():
    open(os.path.join(MS, f), "w").write(t)
print(f"applied {len(E)} edits to {len(texts)} files")
