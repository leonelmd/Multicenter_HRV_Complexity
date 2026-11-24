#!/usr/bin/env python3
"""Maintainer tool: keep the manuscript and this repository in step.

1. Copies figures/<Name>/<Name>.png into the manuscript's figures/ folder under the same
   name (Fig. 1 is exported from Affinity Designer as figures/Fig1_overview/Fig1_overview.pdf).
2. Rebuilds FIGURE_LEGENDS.md from the manuscript captions, which are the single source
   of the legends.

Not needed to reproduce anything. The manuscript is expected next to this repository
(../Multicohort_rcMSE_manuscript) or at $MANUSCRIPT_DIR.

    python scripts/sync_manuscript.py
"""
import os
import re
import shutil

from paths import REPO, FIGURES

MS = os.path.abspath(os.environ.get("MANUSCRIPT_DIR", os.path.join(REPO, "..", "Multicohort_rcMSE_manuscript")))
NAMES = ["Fig1_overview", "Fig2_archetypes", "Fig3_curves_index", "Fig4_age", "Fig5_parameters",
         "Fig6_detector", "Fig7_recording_length", "Fig8_hrv_relationships", "Fig9_clinical",
         "FigS1_hr_normalization", "FigS2_spectral", "FigS3_cruces_context", "FigS4_length",
         "FigS5_ppg_scales", "FigS6_loco", "FigS7_transfer", "FigS8_validation", "FigS9_traces"]
FOLDER = {"FigS6_loco": "FigS6_S7_cross_cohort", "FigS7_transfer": "FigS6_S7_cross_cohort"}


def source(name):
    d = os.path.join(FIGURES, FOLDER.get(name, name))
    for ext in (".pdf", ".png") if name == "Fig1_overview" else (".png",):
        if os.path.exists(os.path.join(d, name + ext)):
            return os.path.join(d, name + ext)
    return None


def captions(tex, prefix):
    out = []
    for n, m in enumerate(re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.S), 1):
        b = m.group(0)
        g = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", b)
        c = re.search(r"\\caption\{(.*)\}\s*\\label", b, re.S)
        lab = re.search(r"\\label\{([^}]+)\}", b)
        out.append((os.path.splitext(g.group(1))[0], f"{prefix}{n}", c.group(1) if c else "",
                    lab.group(1) if lab else ""))
    return out


def plain(s, labels):
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), s)
    s = re.sub(r"\\cite\{[^}]+\}", "[ref]", s)
    s = s.replace("~", " ").replace("\\'e", "é").replace("\\,", " ").replace("--", "–")
    s = re.sub(r"\\(emph|textit)\{([^}]+)\}", r"*\2*", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    for name in NAMES:
        src = source(name)
        if src:
            shutil.copyfile(src, os.path.join(MS, "figures", os.path.basename(src)))
        else:
            print("  no render for", name, "(run reproduce.py first)")

    sec = lambda f: open(os.path.join(MS, "sections", f)).read()
    figs = (captions("".join(sec(f) for f in ("01_introduction.tex", "02_methods.tex",
                                              "03_results.tex", "04_discussion.tex")), "Fig. ")
            + captions(sec("06_supplementary.tex"), "Supplementary Fig. S"))
    labels = {}
    for aux in ("main.aux", "supplementary.aux"):   # section and table numbers, if compiled
        if os.path.exists(os.path.join(MS, aux)):
            for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", open(os.path.join(MS, aux)).read()):
                labels.setdefault(m.group(1), m.group(2))
    labels.update({lab: num.split()[-1] for _, num, _, lab in figs})
    lines = ["# Figure legends", "",
             "Copied from the manuscript captions by `scripts/sync_manuscript.py`; edit the captions",
             "in the manuscript, not here. Each figure is `figures/<name>/<name>.svg`; Suppl. Figs. S6",
             "and S7 share `figures/FigS6_S7_cross_cohort/`.", ""]
    for name, num, cap, _ in figs:
        lines += [f"## {num} — `{name}`", "", plain(cap, labels), ""]
    open(os.path.join(REPO, "FIGURE_LEGENDS.md"), "w").write("\n".join(lines))
    print(f"copied figures to {MS}/figures and wrote FIGURE_LEGENDS.md ({len(figs)} legends)")


if __name__ == "__main__":
    main()
