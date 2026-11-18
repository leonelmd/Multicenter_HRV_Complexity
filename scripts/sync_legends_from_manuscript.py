#!/usr/bin/env python3
"""Rebuild FIGURE_LEGENDS.md from the manuscript captions (the single source of truth).

The manuscript (../Multicohort_rcMSE_manuscript) holds the legends that are submitted;
this script copies each caption into FIGURE_LEGENDS.md under the repository's own figure
names, with a table mapping repository files to manuscript figure numbers. Then run
build_legends_txt.py to refresh figures/legends.txt.

Run: python scripts/sync_legends_from_manuscript.py
"""
import os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.abspath(os.path.join(ROOT, "..", "Multicohort_rcMSE_manuscript"))
OUT = os.path.join(ROOT, "FIGURE_LEGENDS.md")

# repository heading, repository file, manuscript file
MAP = [("Figure 1", "figures/Figure1/Figure1.pdf", "fig1_overview.pdf"),
       ("Figure 2", "figures/Figure2/Figure2.png", "fig2_archetypes.png"),
       ("Figure 4", "figures/Figure4/Figure4.png", "fig3_curves_index.png"),
       ("Figure 6", "figures/Figure6/Figure6.png", "fig5_age.png"),
       ("Appendix 6", "figures/Appendix/FigureAppendix6.png", "figS_params.png"),
       ("Detector", "figures/FigureDetector/FigureDetector.png", "fig_detector.png"),
       ("Figure 3", "figures/Figure3/Figure3.png", "fig4_recording_length.png"),
       ("Figure 7", "figures/Figure7/Figure7.png", "fig6_hrv_relationships.png"),
       ("Figure 8", "figures/Figure8/Figure8.png", "figS_clinical.png"),
       ("Appendix 5", "figures/Appendix/FigureAppendix5.png", "figS_hrnorm.png"),
       ("Appendix 4", "figures/Appendix/FigureAppendix4.png", "figS_spectral.png"),
       ("Appendix 7", "figures/Appendix/FigureAppendix7.png", "figS_cruces_context.png"),
       ("Appendix 3", "figures/Appendix/FigureAppendix3.png", "figS_length.png"),
       ("Appendix 2", "figures/Appendix/FigureAppendix2.png", "figS_ppg_scales.png"),
       ("S6 loco", "figures/Appendix/FigureS6_loco.png", "figS_loco.png"),
       ("S7 transfer", "figures/Appendix/FigureS7_transfer.png", "figS_generalization.png"),
       ("Validation", "figures/Validation/FigureValidation.png", "figS_validation.png"),
       ("S traces", "figures/Appendix/FigureS_traces.png", "figS_traces.png")]


def figures(tex, prefix):
    out, n = [], 0
    for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.S):
        b = m.group(0)
        g = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", b)
        c = re.search(r"\\caption\{(.*)\}\s*\\label", b, re.S)
        lab = re.search(r"\\label\{([^}]+)\}", b)
        n += 1
        out.append((g.group(1), f"{prefix}{n}", c.group(1) if c else "", lab.group(1) if lab else ""))
    return out


def plain(s, labels):
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), s)
    s = re.sub(r"\\cite\{[^}]+\}", "[ref]", s)
    s = s.replace("~", " ").replace("\\'e", "é").replace("\\,", " ").replace("--", "–")
    s = re.sub(r"\\(emph|textit)\{([^}]+)\}", r"*\2*", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    main_tex = "".join(open(os.path.join(MS, "sections", f)).read()
                       for f in ("01_introduction.tex", "02_methods.tex", "03_results.tex", "04_discussion.tex"))
    supp_tex = open(os.path.join(MS, "sections", "06_supplementary.tex")).read()
    figs = figures(main_tex, "Fig. ") + figures(supp_tex, "Supplementary Fig. S")
    labels = {lab: num.split()[-1] for _, num, _, lab in figs}
    by_file = {f: (num, cap) for f, num, cap, _ in figs}
    lines = ["# Figure legends", "",
             "Generated from the manuscript captions by `scripts/sync_legends_from_manuscript.py`;",
             "edit the captions in the manuscript, not here.", "",
             "| Repository file | Manuscript |", "|---|---|"]
    for head, rf, mf in MAP:
        lines.append(f"| `{rf}` | {by_file.get(mf, ('not in manuscript',))[0]} |")
    lines.append("")
    for head, rf, mf in MAP:
        if mf not in by_file:
            continue
        num, cap = by_file[mf]
        first = plain(cap, labels).split(". ")[0]
        lines += [f"## {head} — {first} ({num})", "", plain(cap, labels), ""]
    open(OUT, "w").write("\n".join(lines))
    print(f"wrote {OUT} ({len(figs)} manuscript figures)")


if __name__ == "__main__":
    main()
